"""
SigLIP-based zero-shot emotion classifier for per-frame analysis.

Replaces the previous ResNet-18/FER2013 approach (which had random weights
and produced meaningless output) with SigLIP zero-shot classification
against 8 Plutchik emotion prompts.

SigLIP compares an image to text descriptions and returns similarity logits.
We define one prompt per emotion, run them together, and softmax the logits
to get a probability distribution over the 8 emotions.
"""
from __future__ import annotations

import warnings
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Suppress transformers logging noise on first load
warnings.filterwarnings("ignore", message=".*pretrained.*deprecated.*")
warnings.filterwarnings("ignore", message=".*Arguments other than.*deprecated.*")


# ------------------------------------------------------------------
# Emotion prompts for SigLIP zero-shot classification
# ------------------------------------------------------------------
# These are tuned to elicit strong, distinct responses from SigLIP.
# Each prompt describes a person expressing the target emotion.
EMOTION_PROMPTS: Dict[str, str] = {
    "joy":        "a joyful, happy, delighted person smiling",
    "sadness":    "a sad, crying, sorrowful person looking down",
    "anger":      "an angry, furious, enraged person shouting",
    "fear":       "a fearful, scared, terrified person trembling",
    "surprise":   "a surprised, astonished, shocked person with wide eyes",
    "disgust":    "a disgusted, revolted, nauseated person grimacing",
    "trust":      "a trusting, calm, peaceful, relaxed person",
    "anticipation": "an excited, eager, anticipating person looking forward",
}


class EmotionNet:
    """Zero-shot 8-emotion classifier using SigLIP.

    Loads the model lazily on first call to keep import-time fast.
    """

    _MODEL = None
    _PROCESSOR = None
    _DEVICE = None

    def __init__(self, device: str = "cpu"):
        self.device = device

    # ------------------------------------------------------------------
    # Lazy loading (shared across instances)
    # ------------------------------------------------------------------
    def _load(self):
        if EmotionNet._MODEL is not None:
            return
        from transformers import AutoProcessor, AutoModel
        model_name = "google/siglip-base-patch16-224"
        EmotionNet._PROCESSOR = AutoProcessor.from_pretrained(model_name)
        EmotionNet._MODEL = AutoModel.from_pretrained(model_name)
        EmotionNet._DEVICE = self.device
        EmotionNet._MODEL.to(self.device).eval()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_emotion(self, rgb_img: np.ndarray) -> Dict[str, float]:
        """
        rgb_img – H×W×3 uint8 array
        Returns a dict {emotion: probability in [0,1]} for the 8 emotions.
        """
        self._load()

        pil_img = Image.fromarray(rgb_img) if isinstance(rgb_img, np.ndarray) else rgb_img

        # Process image
        image_inputs = EmotionNet._PROCESSOR(
            images=pil_img,
            return_tensors="pt",
        ).to(EmotionNet._DEVICE)

        # Process all 8 emotion prompts
        text_inputs = EmotionNet._PROCESSOR(
            text=list(EMOTION_PROMPTS.values()),
            return_tensors="pt",
            padding=True,
        ).to(EmotionNet._DEVICE)

        # SigLIP forward → logits_per_image: [1, 8]
        outputs = EmotionNet._MODEL(**image_inputs, **text_inputs)
        logits = outputs.logits_per_image[0]  # [8]

        # Softmax to probabilities
        probs = F.softmax(logits, dim=0).cpu().numpy()

        return {
            emotion: round(float(probs[i]), 4)
            for i, emotion in enumerate(EMOTION_PROMPTS.keys())
        }
