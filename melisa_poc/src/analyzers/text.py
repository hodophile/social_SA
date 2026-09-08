"""English social-media text sentiment analyzer (RoBERTa)."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
from scipy.special import softmax
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from src.config import DEFAULT_TEXT_MODEL, TEXT_CHUNK_SIZE, TEXT_CHUNK_STRIDE, TEXT_MAX_LENGTH
from src.schemas import SentimentEvidence, SentimentLabel

logger = logging.getLogger(__name__)

# Canonical label order used for score = P(pos) - P(neg).
_LABEL_KEYS: tuple[SentimentLabel, ...] = ("negative", "neutral", "positive")


class TextSentimentAnalyzer:
    """Lazy-loaded Twitter-RoBERTa sentiment classifier for English text.

    Score convention (POC, not client business scoring):
        score = positive_probability - negative_probability  ∈ [-1, +1]

    Long inputs (transcripts) are split into overlapping token chunks and
    probability distributions are aggregated with length weighting — the full
    text is never silently discarded beyond documented chunking.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_TEXT_MODEL,
        device: Optional[str] = None,
        max_length: int = TEXT_MAX_LENGTH,
        chunk_size: int = TEXT_CHUNK_SIZE,
        chunk_stride: int = TEXT_CHUNK_STRIDE,
    ) -> None:
        self.model_name = model_name
        self._device_preference = device
        self.max_length = max_length
        self.chunk_size = min(chunk_size, max_length)
        self.chunk_stride = max(1, chunk_stride)
        self._tokenizer = None
        self._model = None
        self._id2label: dict[int, str] = {}
        self._device: Optional[torch.device] = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _resolve_device(self) -> torch.device:
        # Always CPU. ZeroGPU advertises a GPU as available; text must not follow that.
        return torch.device("cpu")

    def _normalize_label(self, raw: str) -> SentimentLabel:
        key = raw.strip().lower()
        # Some checkpoints expose LABEL_0 / negative / Negative.
        aliases = {
            "label_0": "negative",
            "label_1": "neutral",
            "label_2": "positive",
            "neg": "negative",
            "neu": "neutral",
            "pos": "positive",
            "negative": "negative",
            "neutral": "neutral",
            "positive": "positive",
        }
        if key not in aliases:
            raise ValueError(f"Unexpected sentiment label from model: {raw!r}")
        return aliases[key]  # type: ignore[return-value]

    def load(self) -> None:
        """Download (if needed) and load tokenizer + model onto device."""
        if self.is_loaded:
            return

        logger.info("Loading text sentiment model: %s", self.model_name)
        self._device = torch.device("cpu")

        # CardiffNLP checkpoints commonly emit an unused-pooler warning; keep CLI quiet.
        from transformers.utils import logging as hf_logging

        prev_verbosity = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            config = AutoConfig.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
            )
        finally:
            hf_logging.set_verbosity(prev_verbosity)

        self._model.to("cpu")
        self._model.eval()

        self._id2label = {
            int(i): self._normalize_label(str(label))
            for i, label in config.id2label.items()
        }
        logger.info(
            "Text sentiment model ready on %s (labels=%s)",
            self._device,
            self._id2label,
        )

    @staticmethod
    def validate_text(text: object) -> str:
        """Require a non-blank string. Raises ValueError otherwise."""
        if not isinstance(text, str):
            raise ValueError("text must be a string")
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("text must be a non-blank string")
        return cleaned

    def analyze(self, text: object) -> SentimentEvidence:
        """Run inference and return structured modality evidence."""
        cleaned = self.validate_text(text)
        self.load()

        assert self._tokenizer is not None
        assert self._model is not None
        assert self._device is not None

        token_ids = self._tokenizer.encode(cleaned, add_special_tokens=False)
        if len(token_ids) <= self.max_length - 2:
            return self._analyze_text_once(cleaned, extra_details=None)

        chunks = self._chunk_token_ids(token_ids)
        chunk_results: list[tuple[SentimentEvidence, int]] = []
        for chunk_ids in chunks:
            chunk_text = self._tokenizer.decode(chunk_ids, skip_special_tokens=True).strip()
            if not chunk_text:
                continue
            evidence = self._analyze_text_once(
                chunk_text,
                extra_details={"partial": True},
            )
            chunk_results.append((evidence, len(chunk_ids)))

        if not chunk_results:
            raise ValueError("text produced no scorable chunks after tokenization")
        return self._aggregate_chunk_results(chunk_results, total_tokens=len(token_ids))

    def _chunk_token_ids(self, token_ids: list[int]) -> list[list[int]]:
        """Split long token sequences into overlapping chunks (no silent drop)."""
        if not token_ids:
            return []
        usable = self.chunk_size
        stride = min(self.chunk_stride, usable)
        chunks: list[list[int]] = []
        start = 0
        n = len(token_ids)
        while start < n:
            end = min(start + usable, n)
            chunks.append(token_ids[start:end])
            if end >= n:
                break
            start = max(0, end - stride)
            if start >= end:
                start = end
        return chunks

    def _analyze_text_once(
        self,
        cleaned: str,
        *,
        extra_details: Optional[dict],
    ) -> SentimentEvidence:
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._device is not None

        encoded = self._tokenizer(
            cleaned,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        encoded = {k: v.to("cpu") for k, v in encoded.items()}

        with torch.no_grad():
            outputs = self._model(**encoded)
            logits = outputs.logits.detach().cpu().numpy()[0]

        probs = softmax(logits)
        probability_map: dict[str, float] = {
            self._id2label[i]: float(probs[i]) for i in range(len(probs))
        }

        for key in _LABEL_KEYS:
            probability_map.setdefault(key, 0.0)

        positive_p = probability_map["positive"]
        negative_p = probability_map["negative"]
        score = float(positive_p - negative_p)

        label = max(_LABEL_KEYS, key=lambda k: probability_map[k])
        confidence = float(probability_map[label])

        details: dict = {"device": "cpu"}
        if extra_details:
            details.update(extra_details)

        return SentimentEvidence(
            label=label,
            score=float(np.clip(score, -1.0, 1.0)),
            confidence=confidence,
            probabilities={k: float(probability_map[k]) for k in _LABEL_KEYS},
            model=self.model_name,
            details=details,
        )

    def _aggregate_chunk_results(
        self,
        chunk_results: list[tuple[SentimentEvidence, int]],
        *,
        total_tokens: int,
    ) -> SentimentEvidence:
        """Length-weighted average of chunk probability distributions."""
        if not chunk_results:
            raise ValueError("No chunks to aggregate")
        if len(chunk_results) == 1:
            only, _ = chunk_results[0]
            return only.model_copy(
                update={
                    "details": {
                        **(only.details or {}),
                        "chunking": {
                            "chunks": 1,
                            "total_tokens": total_tokens,
                            "method": "single_chunk",
                        },
                    },
                },
            )

        weight_sum = float(sum(max(w, 1) for _, w in chunk_results))
        agg = {k: 0.0 for k in _LABEL_KEYS}
        for evidence, weight in chunk_results:
            probs = evidence.probabilities or {}
            w = float(max(weight, 1)) / weight_sum
            for key in _LABEL_KEYS:
                agg[key] += w * float(probs.get(key, 0.0))

        total = sum(agg.values()) or 1.0
        probability_map = {k: float(agg[k] / total) for k in _LABEL_KEYS}
        label = max(_LABEL_KEYS, key=lambda k: probability_map[k])
        confidence = float(probability_map[label])
        score = float(probability_map["positive"] - probability_map["negative"])

        return SentimentEvidence(
            label=label,
            score=float(np.clip(score, -1.0, 1.0)),
            confidence=confidence,
            probabilities=probability_map,
            model=self.model_name,
            details={
                "device": "cpu",
                "chunking": {
                    "chunks": len(chunk_results),
                    "total_tokens": total_tokens,
                    "chunk_size": self.chunk_size,
                    "chunk_stride": self.chunk_stride,
                    "method": "length_weighted_probability_average",
                    "note": (
                        "Long transcript/text was chunked for RoBERTa's 512-token limit; "
                        "full text is preserved separately from this aggregate."
                    ),
                },
            },
        )
