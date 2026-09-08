"""
Maps the RoBERTa 3‑class sentiment (used by Melisa for the caption) to an
8‑emotion distribution.  Feel free to replace this with a call to the same
OpenRouter emotion endpoint that the audio branch uses – the interface is the
same (returns a dict {emotion: prob}).
"""
from melisa_poc.src.schemas import SentimentEvidence

EMO_MAP = {
    "positive": {"joy": 0.6, "trust": 0.2, "anticipation": 0.2},
    "neutral":  {"joy": 0.2, "trust": 0.5, "anticipation": 0.3},
    "negative": {"sadness": 0.4, "fear": 0.3, "anger": 0.2, "disgust": 0.1},
}

def caption_to_emotion(caption_sentiment: SentimentEvidence) -> dict:
    base = EMO_MAP.get(caption_sentiment.label, {"joy": 0.33, "trust": 0.33, "anticipation": 0.33})
    scaled = {k: v * float(caption_sentiment.confidence) for k, v in base.items()}
    total = sum(scaled.values())
    if total > 0:
        for k in scaled:
            scaled[k] /= total
    return scaled
