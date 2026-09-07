"""Zero-shot visual sentiment evidence via SigLIP 2 (not a trained sentiment classifier)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import spaces
import torch
from PIL import Image, UnidentifiedImageError
from scipy.special import softmax

from src.config import DEFAULT_VISUAL_MODEL, DEFAULT_VISUAL_PROMPTS
from src.schemas import SentimentEvidence, SentimentLabel

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]
_LABEL_KEYS: tuple[SentimentLabel, ...] = ("negative", "neutral", "positive")


def _siglip_gpu_duration(input_items: dict[str, Any]) -> int:
    """Declare a tight ZeroGPU duration so short SigLIP calls are not billed as 60s."""
    pixel_values = input_items.get("pixel_values")
    batch = 1
    if pixel_values is not None and hasattr(pixel_values, "shape") and pixel_values.shape:
        batch = int(pixel_values.shape[0])
    return int(min(55, max(15, 10 + batch * 3)))


@spaces.GPU(duration=_siglip_gpu_duration)
def _siglip_gpu_forward(input_items: dict[str, Any]) -> np.ndarray:
    """SigLIP 2 forward only. Returns CPU logits; never moves RoBERTa or Whisper.

    Off ZeroGPU, ``spaces.GPU`` is a no-op so this runs in-process on the analyzer device.
    """
    model = VisualSentimentAnalyzer._gpu_model
    device = VisualSentimentAnalyzer._gpu_device
    if model is None or device is None:
        raise RuntimeError("SigLIP 2 model is not loaded")
    tensors = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in input_items.items()
    }
    with torch.no_grad():
        outputs = model(**tensors)
        return outputs.logits_per_image.detach().float().cpu().numpy()


class VisualSentimentAnalyzer:
    """Lazy SigLIP 2 zero-shot scorer against configurable sentiment concept prompts.

    This is **not** a fine-tuned sentiment classifier. Labels come from comparing
    the image to short English concept descriptions, then mapping the highest-
    scoring concept to positive / neutral / negative.
    """

    # Class-level handles so the ZeroGPU worker can run forward without pickling ``self``.
    _gpu_model: Any = None
    _gpu_device: Optional[torch.device] = None

    def __init__(
        self,
        model_name: str = DEFAULT_VISUAL_MODEL,
        prompts: Optional[Mapping[str, str]] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.prompts = dict(prompts or DEFAULT_VISUAL_PROMPTS)
        self._device_preference = device
        self._processor = None
        self._model = None
        self._device: Optional[torch.device] = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _resolve_device(self) -> torch.device:
        if self._device_preference:
            return torch.device(self._device_preference)
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def load(self) -> None:
        if self.is_loaded:
            VisualSentimentAnalyzer._gpu_model = self._model
            VisualSentimentAnalyzer._gpu_device = self._device
            return

        from transformers import AutoModel, AutoProcessor
        from transformers.utils import logging as hf_logging

        logger.info("Loading visual zero-shot model: %s", self.model_name)
        self._device = self._resolve_device()

        prev = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()
        try:
            self._processor = AutoProcessor.from_pretrained(self.model_name)
            # CPU-safe load: no device_map="auto" (avoids accelerate requirement).
            # Do not use torch.compile (unsupported on Hugging Face ZeroGPU).
            self._model = AutoModel.from_pretrained(self.model_name)
        finally:
            hf_logging.set_verbosity(prev)

        self._model.to(self._device)
        self._model.eval()
        VisualSentimentAnalyzer._gpu_model = self._model
        VisualSentimentAnalyzer._gpu_device = self._device
        logger.info("Visual model ready on %s", self._device)

    @staticmethod
    def load_image(path: PathLike) -> Image.Image:
        """Open a local image as RGB. Raises FileNotFoundError / ValueError."""
        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(f"Image not found: {file_path}")
        try:
            with Image.open(file_path) as img:
                return img.convert("RGB")
        except UnidentifiedImageError as exc:
            raise ValueError(f"Unsupported or corrupt image file: {file_path}") from exc
        except OSError as exc:
            raise ValueError(f"Failed to open image file: {file_path} ({exc})") from exc

    def _evidence_from_logits(self, logits: np.ndarray) -> SentimentEvidence:
        """Map a single image's concept logits to the existing evidence schema."""
        ordered_labels: list[SentimentLabel] = list(_LABEL_KEYS)
        probs = softmax(logits)
        raw_similarities = 1.0 / (1.0 + np.exp(-logits))
        probability_map = {
            label: float(probs[i]) for i, label in enumerate(ordered_labels)
        }
        similarity_map = {
            label: float(raw_similarities[i]) for i, label in enumerate(ordered_labels)
        }
        score = float(probability_map["positive"] - probability_map["negative"])
        label = max(ordered_labels, key=lambda k: probability_map[k])
        confidence = float(probability_map[label])
        return SentimentEvidence(
            label=label,
            score=float(np.clip(score, -1.0, 1.0)),
            confidence=confidence,
            probabilities={k: float(probability_map[k]) for k in ordered_labels},
            model=self.model_name,
            details={
                "method": "zero-shot concept scoring",
                "prompts": {k: self.prompts[k] for k in ordered_labels},
                "raw_similarities": similarity_map,
                "device": str(self._device),
            },
        )

    def analyze_images(self, images: Sequence[Image.Image]) -> list[SentimentEvidence]:
        """Score one or more PIL images in a single SigLIP forward (one ZeroGPU entry)."""
        if not images:
            return []
        self.load()
        assert self._processor is not None
        assert self._model is not None
        assert self._device is not None

        ordered_labels: list[SentimentLabel] = list(_LABEL_KEYS)
        texts = [self.prompts[label] for label in ordered_labels]
        # SigLIP 2 training used padding="max_length", max_length=64.
        inputs = self._processor(
            text=texts,
            images=list(images),
            padding="max_length",
            max_length=64,
            return_tensors="pt",
        )
        cpu_inputs = {
            key: value.detach().cpu() if torch.is_tensor(value) else value
            for key, value in inputs.items()
        }
        logits_batch = _siglip_gpu_forward(cpu_inputs)
        if logits_batch.ndim == 1:
            logits_batch = np.expand_dims(logits_batch, axis=0)
        return [self._evidence_from_logits(row) for row in logits_batch]

    def analyze_image(self, image: Image.Image) -> SentimentEvidence:
        """Score an in-memory PIL image against sentiment concept prompts."""
        results = self.analyze_images([image])
        if not results:
            raise RuntimeError("SigLIP 2 produced no visual evidence")
        return results[0]

    def analyze_path(self, path: PathLike) -> SentimentEvidence:
        return self.analyze_image(self.load_image(path))
