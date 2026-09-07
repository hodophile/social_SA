"""
Orchestrator that builds the three modalities (caption, video, audio),
runs the existing Melisa confidence‑weighted late fusion, and returns a
JSON blob that matches the format used by the current Qwen + OpenRouter
pipeline (execution_time_seconds, understanding, risk, action,
emotion_analysis).
"""
from __future__ import annotations

from dataclasses import field
import time
import json
from pathlib import Path
from typing import Optional, Union
from pydantic.dataclasses import dataclass

import requests

from melisa_poc.src.schemas import (
    SentimentEvidence,
    SpeechAnalysisResult,
    VideoDiagnostics,
    # SocialMediaPost,
)
from melisa_poc.src.analyzers.text import TextSentimentAnalyzer
from melisa_poc.src.analyzers.image import ImageAnalyzer
from melisa_poc.src.analyzers.audio import AudioAnalyzer
from melisa_poc.src.fusion import fuse_modalities, DEFAULT_FUSION
from melisa_temporal_emotion.video_analyzer import VideoAnalyzer
from melisa_temporal_emotion.text_analyzer import caption_to_emotion
from melisa_temporal_emotion.models import EmotionNet

@dataclass
class VideoAnalysisBundle:
    """Pre-activity multimodal evidence for one video (pipeline wraps into ActivityAnalysisResult)."""

    visual: Optional[SentimentEvidence]
    ocr: Optional[SentimentEvidence]
    ocr_text: Optional[str]
    speech: Optional[SentimentEvidence]
    transcript: Optional[str]
    speech_result: Optional[SpeechAnalysisResult]
    diagnostics: VideoDiagnostics
    warnings: list[str] = field(default_factory=list)
    overall: Optional[SentimentEvidence] = None

@dataclass
class SocialMediaPost:
    """
    Raw social-media post.

    A post may contain text, an image, a video, or any
    combination of these.
    """

    post_id: str
    text: Optional[str] = None
    image_path: Optional[Path] = None
    video_path: Optional[Path] = None

PathLike = Union[str, Path]
EMOTION_LABELS = [
    "joy",
    "sadness",
    "anger",
    "fear",
    "surprise",
    "disgust",
    "trust",
    "anticipation",
]

VALENCE_MAP = {
    "joy": +1.0,
    "trust": +0.5,
    "anticipation": +0.3,
    "surprise": 0.0,
    "fear": -0.5,
    "anger": -0.8,
    "sadness": -1.0,
    "disgust": -0.8,
}


class TemporalEmotionPipeline:
    def __init__(
        self,
        *,
        video_analyzer: Optional[VideoAnalyzer] = None,
        audio_analyzer: Optional[AudioAnalyzer] = None,
        text_analyzer: Optional[TextSentimentAnalyzer] = None,
        openrouter_api_key: Optional[str] = None,
        openrouter_model: str = "openai/gpt-4o-mini",
    ):
        self._video = video_analyzer or VideoAnalyzer()
        self._audio = audio_analyzer or AudioAnalyzer(
            ffmpeg_path=None,
            language="en",
            compute_type="int8",
            model="base.en",
        )
        self._text = text_analyzer or TextSentimentAnalyzer()
        self._openrouter_key = openrouter_api_key
        self._openrouter_model = openrouter_model

    # ------------------------------------------------------------------
    # Helper: OpenRouter emotion analysis (same as in app.py)
    # ------------------------------------------------------------------
    def _openrouter_emotion(self, text: str) -> dict:
        if not self._openrouter_key:
            return {"skipped": True, "reason": "no OPENROUTER_API_KEY"}
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._openrouter_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._openrouter_model,
            "messages": [
                {
                    "role": "user",
                    "content": self._make_prompt(text),
                }
            ],
            "temperature": 0,
        }
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"]
        start = raw.find("{")
        end = raw.rfind("}") + 1
        data = json.loads(raw[start:end])
        return {
            "primary_emotion": data.get("primary_emotion"),
            "secondary_emotion": data.get("secondary_emotion"),
            "emotion_scores": data.get("emotion_scores", {}),
            "valence": data.get("valence", 0.0),
            "confidence": data.get("confidence", 0.5),
            "rationale": data.get("rationale", ""),
        }

    def _make_prompt(self, text: str) -> str:
        return (
            "Based ONLY on the following text, identify the emotion conveyed. "
            "Score each of these emotions from 0.0 to 1.0: joy, sadness, anger, fear, surprise, disgust, trust, anticipation. "
            "Return ONLY JSON:\n"
            "{\n"
            '  "primary_emotion": "<one of the eight>",\n'
            '  "secondary_emotion": "<one of the eight or null>",\n'
            '  "emotion_scores": {"joy":0.0,"sadness":0.0,"anger":0.0,"fear":0.0,"surprise":0.0,"disgust":0.0,"trust":0.0,"anticipation":0.0},\n'
            '  "valence": <-1.0 to +1.0>,\n'
            '  "confidence": 0.0-1.0,\n'
            '  "rationale": "one short sentence"\n'
            "}\n\nText:\n"
            f"{text}"
        )

    # ------------------------------------------------------------------
    # Main analyse method – matches the signature used by the Gradio/Streamlit front‑end
    # ------------------------------------------------------------------
    def analyze(
        self,
        text: Optional[str] = None,
        image_path: Optional[PathLike] = None,
        video_path: Optional[PathLike] = None,
        sample_fps: float = 1.0,
    ) -> dict:
        t0 = time.perf_counter()

        post = SocialMediaPost(
            post_id="temp-post",
            text=text,
            image_path=Path(image_path) if image_path else None,
            video_path=Path(video_path) if video_path else None,
        )

        # ------------------------------------------------------------------
        # 1️⃣ Caption → 8‑emotion distribution (heuristic mapping)
        # ------------------------------------------------------------------
        caption_emotion = {}
        if post.text:
            cap_sent = self._text.analyze(post.text)
            caption_emotion = caption_to_emotion(cap_sent)
        # Build a SentimentEvidence for the fusion step
        caption_evidence = SentimentEvidence(
            label="placeholder",
            score=0.0,
            confidence=1.0,
            probabilities=caption_emotion,
            model="caption_emotion_heuristic",
            details={},
        )

        # ------------------------------------------------------------------
        # 2️⃣ Video → per‑frame 8‑emotion → temporal aggregation
        # ------------------------------------------------------------------
        video_evidence = SentimentEvidence(
            label="placeholder",
            score=0.0,
            confidence=1.0,
            probabilities={e: 0.0 for e in EMOTION_LABELS},
            model="video_temporal_emotion",
            details={},
        )
        if post.video_path:
            video_bundle: VideoAnalysisBundle = self._video.analyze(
                post.video_path,
                caption_sentiment=caption_evidence,
            )
            frame_emotions = getattr(video_bundle, "frame_emotions", [])
            if frame_emotions:
                video_dist = temporal_weighted_average(
                    frame_emotions,
                    half_life_seconds=2.0,
                    fps=getattr(video_bundle.diagnostics, "sampling_fps", 1.0),
                )
                video_evidence = SentimentEvidence(
                    label="placeholder",
                    score=0.0,
                    confidence=1.0,
                    probabilities=video_dist,
                    model="video_temporal_emotion",
                    details={},
                )

        # ------------------------------------------------------------------
        # 3️⃣ Audio transcript → 8‑emotion via OpenRouter (fallback to RoBERTa)
        # ------------------------------------------------------------------
        audio_evidence = SentimentEvidence(
            label="placeholder",
            score=0.0,
            confidence=1.0,
            probabilities={e: 0.0 for e in EMOTION_LABELS},
            model="audio_emotion",
            details={},
        )
        if post.video_path:
            speech_result = self._audio.analyze(post.video_path)
            if speech_result.transcript and self._openrouter_key:
                emo = self._openrouter_emotion(speech_result.transcript)
                if not emo.get("skipped"):
                    audio_evidence = SentimentEvidence(
                        label="placeholder",
                        score=0.0,
                        confidence=emo["confidence"],
                        probabilities=emo["emotion_scores"],
                        model="openrouter_emotion",
                        details={"valence": emo["valence"], "rationale": emo["rationale"]},
                    )
            # fallback to RoBERTa if no API key or error
            elif speech_result.transcript and self._text is not None:
                txt_sent = self._text.analyze(speech_result.transcript)
                audio_evidence = SentimentEvidence(
                    label="placeholder",
                    score=0.0,
                    confidence=txt_sent.confidence,
                    probabilities=caption_to_emotion(txt_sent),   # reuse heuristic
                    model="audio_emotion_roberta",
                    details={},
                )

        # ------------------------------------------------------------------
        # 4️⃣ Final fusion (confidence‑weighted late fusion – exactly Melisa’s)
        # ------------------------------------------------------------------
        fused = fuse_modalities(
            {
                "text":   caption_evidence,
                "visual": video_evidence,
                "speech": audio_evidence,
            },
            config=DEFAULT_FUSION,
        )
        overall = fused.overall   # SentimentEvidence with .probabilities = 8‑emotion dist

        exec_time = round(time.perf_counter() - t0, 3)

        # ------------------------------------------------------------------
        # Build the final JSON in the exact format used by the Qwen setup
        # ------------------------------------------------------------------
        understanding = {
            "content": post.text or "No caption provided.",
            "tone": "neutral",                     # placeholder – could be derived from valence
            "intent": "informational",             # placeholder
            "entities": [],
            "visual_description": (
                f"{len(getattr(video_bundle, 'frame_paths', []))} video frames processed"
                if post.video_path else "No video"
            ),
            "audio_description": (
                "Audio present and transcribed"
                if post.video_path and getattr(self._audio, "_whisper_model", None)
                else "No audio"
            ),
            "potential_harm": [],
        }

        # ---- risk (same as before – we keep the harmless default) ----
        risk = {
            "score": 0.0,
            "labels": [],
            "probabilities": {},
        }

        # ---- action – harmless by default (you can plug in your own thresholds) ----
        action = "ALLOW"

        # ---- emotion analysis block (matches the Qwen+OpenRouter output) ----
        # Extract the fused 8‑emotion distribution from overall.probabilities
        emo_dist = overall.probabilities or {e: 0.0 for e in EMOTION_LABELS}
        # Find primary emotion (highest probability)
        primary = max(emo_dist.items(), key=lambda kv: kv[1])[0] if emo_dist else "joy"
        # Valence: we approximate it as the weighted sum of emotion → valence mapping
        valence = sum(emo_dist[e] * VALENCE_MAP[e] for e in EMOTION_LABELS)
        confidence = overall.confidence or 0.0
        primary_score = emo_dist.get(primary, 0.0) * confidence

        emotion_analysis = {
            "emotion_model": "temporal_emotion_net + caption_heuristic + openrouter",
            "primary_emotion": primary,
            "secondary_emotion": None,   # we could compute second‑highest if desired
            "emotion_scores": emo_dist,          # raw 0‑1 scores (not normalised)
            "emotion_distribution": {k: round(v, 3) for k, v in emo_dist.items()},
            "valence": round(valence, 3),
            "confidence": confidence,
            "primary_emotion_score": round(primary_score, 3),
            "rationale": (
                "Temporal‑weighted average of per‑frame emotion net output, "
                "caption‑heuristic mapping, and audio transcript analysed via OpenRouter."
            ),
        }

        return {
            "execution_time_seconds": exec_time,
            "post_id": post.post_id,
            "understanding": understanding,
            "risk": risk,
            "action": action,
            "emotion_analysis": emotion_analysis,
        }
