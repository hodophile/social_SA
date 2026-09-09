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
# EMOTION_PROMPTS: Dict[str, str] = {
#     "joy":        "a joyful, happy, delighted person smiling",
#     "sadness":    "a sad, crying, sorrowful person looking down",
#     "anger":      "an angry, furious, enraged person shouting",
#     "fear":       "a fearful, scared, terrified person trembling",
#     "surprise":   "a surprised, astonished, shocked person with wide eyes",
#     "disgust":    "a disgusted, revolted, nauseated person grimacing",
#     "trust":      "a trusting, calm, peaceful, relaxed person",
#     "anticipation": "an excited, eager, anticipating person looking forward",
# }


EMOTION_PROMPTS = {
    "joy": (
        "A social-media image or video conveying strong joy, happiness, and positive energy. "
        "The emotion may be expressed through people smiling, laughing, celebrating, playing, "
        "hugging, spending time together, enjoying food or activities, or through positive scenes "
        "such as beautiful nature, colorful surroundings, pets, achievements, festivals, weddings, "
        "success, friendship, family moments, or uplifting events. Use bright and warm visual tones, "
        "lively activity, open body language, energetic movement, pleasant surroundings, and an overall "
        "sense of celebration, pleasure, connection, optimism, and emotional warmth. The scene does "
        "not require a person; objects, environments, events, and visual context can communicate joy."
    ),

    "sadness": (
        "A social-media image or video conveying sadness, sorrow, grief, loneliness, disappointment, "
        "or emotional pain. The emotion may be expressed through crying or distressed people, but can "
        "also appear through empty or abandoned places, separation, loss, funerals, memorials, damaged "
        "homes, lonely individuals, rainy or gloomy environments, neglected spaces, fading memories, "
        "or situations involving failure or hardship. Use subdued colors, low-energy compositions, "
        "downcast or isolated subjects, empty spaces, stillness, and melancholic atmosphere where "
        "appropriate. The overall context should communicate emotional heaviness, loss, loneliness, "
        "helplessness, or sorrow even when no person is visible."
    ),

    "anger": (
        "A social-media image or video conveying anger, rage, outrage, hostility, frustration, or "
        "strong opposition. The emotion may be expressed through visibly angry people, arguments, "
        "confrontations, protests, shouting, aggressive gestures, or tense crowds, but can also be "
        "communicated through scenes of injustice, destruction, violence, corruption, unfair treatment, "
        "political or social conflict, property damage, or other situations that provoke outrage. "
        "Look for tense body language, confrontational interactions, clenched fists, aggressive gestures, "
        "chaotic movement, harsh visual composition, warning signs, damaged objects, or visibly hostile "
        "situations. The emotion should feel intense, confrontational, and negative rather than merely "
        "serious or concerned."
    ),

    "fear": (
        "A social-media image or video conveying fear, anxiety, panic, danger, threat, or insecurity. "
        "The emotion may be expressed through frightened or fleeing people, but can also be communicated "
        "by dangerous environments, natural disasters, accidents, fires, storms, floods, war, threatening "
        "animals, unsafe situations, darkness, destruction, or scenes suggesting imminent danger. "
        "Use visual cues such as people running or hiding, defensive body language, chaotic situations, "
        "dark or threatening surroundings, emergency conditions, destruction, uncertainty, or isolation. "
        "The overall context should suggest vulnerability, danger, alarm, or a desire to escape or seek safety, "
        "even when no person is present."
    ),

    "surprise": (
        "A social-media image or video conveying surprise, astonishment, shock, amazement, or an unexpected "
        "discovery. The emotion may be expressed through people with wide eyes, open mouths, startled reactions, "
        "or sudden gestures, but can also come from unexpected events, unusual objects, dramatic transformations, "
        "rare natural phenomena, unexpected outcomes, sudden accidents, extraordinary achievements, shocking news, "
        "or visually unusual situations. Emphasize a clear contrast between what would normally be expected and "
        "what is actually happening. The scene should communicate an immediate sense of something unexpected, "
        "unbelievable, or astonishing, whether or not people are visible."
    ),

    "disgust": (
        "A social-media image or video conveying disgust, revulsion, nausea, repulsion, or strong aversion. "
        "The emotion may be expressed through people grimacing, covering their noses, turning away, or reacting "
        "negatively, but can also be communicated directly through unpleasant or contaminated scenes such as "
        "rotting food, garbage, pollution, sewage, unhygienic conditions, spoiled substances, infestation, "
        "severe neglect, disturbing contamination, or visibly revolting environments. Use visual cues such as "
        "dirty or decaying surroundings, unpleasant textures, contamination, foul-looking substances, people "
        "recoiling, or objects being avoided. The overall scene should communicate a strong desire to reject, "
        "avoid, or distance oneself from something unpleasant."
    ),

    "trust": (
        "A social-media image or video conveying trust, safety, reassurance, confidence, acceptance, peace, "
        "and emotional security. The emotion may be expressed through people helping, supporting, cooperating, "
        "comforting, caring for one another, forming strong relationships, or interacting peacefully, but can "
        "also be communicated through safe and welcoming environments, reliable services, community cooperation, "
        "protective actions, responsible organizations, successful teamwork, or positive relationships between "
        "people and animals. Use relaxed body language, open interactions, supportive gestures, calm environments, "
        "stable compositions, welcoming surroundings, and visual signs of reliability and security. The overall "
        "context should communicate safety, dependability, care, reassurance, or confidence rather than simple "
        "happiness."
    ),

    "anticipation": (
        "A social-media image or video conveying anticipation, eagerness, expectation, hope, curiosity, or "
        "excitement about something that is about to happen. The emotion may be expressed through people waiting, "
        "preparing, looking toward an expected event, gathering before a celebration, watching a countdown, "
        "opening a package, preparing for travel, awaiting results, or getting ready for an important occasion. "
        "It can also be communicated without people through visual clues such as unopened packages, event setups, "
        "stages prepared for an audience, countdowns, starting lines, preparations, announcements, unfinished "
        "reveals, or scenes suggesting an upcoming event. The key characteristic is that something meaningful "
        "or exciting is expected to happen in the near future, creating a sense of curiosity, eagerness, tension, "
        "or hopeful excitement."
    ),
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
