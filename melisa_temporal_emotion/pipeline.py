"""
TemporalEmotionPipeline — Melisa-based pipeline with temporal 8-emotion output.

Modality flow:
  caption  -> RoBERTa sentiment  -> mapped to 8 emotions (heuristic)
  video    -> per-frame 8-emotion CNN -> temporal weighted average
  audio    -> Whisper transcript    -> OpenRouter 8-emotion (or RoBERTa fallback)

The three 8-emotion distributions are fused with the same
confidence-weighted pattern Melisa uses for its late fusion, yielding a
single 8-emotion distribution, valence and confidence.

Output format matches the Qwen2.5-VLM-3B pipeline JSON:
  execution_time_seconds / post_id / understanding / risk / action /
  emotion_analysis
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

import requests

# melisa_poc internal modules import each other as `src.*`, so we must
# expose the melisa_poc directory itself on sys.path.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_MELISA_ROOT = _REPO_ROOT / "melisa_poc"
for _p in (str(_REPO_ROOT), str(_MELISA_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from melisa_poc.src.analyzers.text import TextSentimentAnalyzer
from melisa_poc.src.analyzers.audio import AudioAnalyzer
from melisa_temporal_emotion.video_analyzer import (
    VideoAnalyzer,
    VideoAnalysisBundle,
)
from melisa_temporal_emotion.text_analyzer import caption_to_emotion
from melisa_temporal_emotion.fusion import (
    EMOTION_LABELS,
    fuse_emotion_distributions,
    temporal_weighted_average,
    valence_of,
)


PathLike = Union[str, Path]


@dataclass
class SocialMediaPost:
    """Minimal input record (melisa_poc has no such schema)."""

    post_id: str
    text: Optional[str] = None
    image_path: Optional[Path] = None
    video_path: Optional[Path] = None


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
        self._text = text_analyzer or TextSentimentAnalyzer()
        self._audio = audio_analyzer or AudioAnalyzer(
            whisper_model="base.en",
            compute_type="int8",
            language="en",
            text_analyzer=self._text,
        )
        self._video = video_analyzer or VideoAnalyzer(
            audio_analyzer=self._audio,
            text_analyzer=self._text,
        )
        self._openrouter_key = openrouter_api_key
        self._openrouter_model = openrouter_model

    # ------------------------------------------------------------------
    # OpenRouter emotion analysis over a text (e.g. audio transcript)
    # ------------------------------------------------------------------
    def _openrouter_emotion(self, text: str) -> dict:
        if not self._openrouter_key:
            return {"skipped": True, "reason": "no OPENROUTER_API_KEY"}
        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self._openrouter_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._openrouter_model,
                "messages": [{"role": "user", "content": self._make_prompt(text)}],
                "temperature": 0,
            },
            timeout=60,
        )
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"]
        start, end = raw.find("{"), raw.rfind("}") + 1
        return json.loads(raw[start:end])

    @staticmethod
    def _make_prompt(text: str) -> str:
        return (
            "Based ONLY on the following text, identify the emotion conveyed. "
            "Score each of these emotions from 0.0 to 1.0: "
            + ", ".join(EMOTION_LABELS)
            + ". Return ONLY JSON:\n"
            "{\n"
            '  "primary_emotion": "<one of the eight>",\n'
            '  "secondary_emotion": "<one of the eight or null>",\n'
            '  "emotion_scores": {'
            + ", ".join(f'"{e}": 0.0' for e in EMOTION_LABELS)
            + "},\n"
            '  "valence": <-1.0 to +1.0>,\n'
            '  "confidence": 0.0-1.0,\n'
            '  "rationale": "one short sentence"\n'
            "}\n\nText:\n"
            f"{text}"
        )

    # ------------------------------------------------------------------
    # Modality branches -> (8-emotion distribution, confidence)
    # ------------------------------------------------------------------

    def _caption_branch(self, text: Optional[str]):
        if not text:
            return None
        cap = self._text.analyze(text)
        dist = caption_to_emotion(cap)
        return dist, float(cap.confidence)

    def _video_branch(self, video_path: Path):
        bundle: VideoAnalysisBundle = self._video.analyze(video_path)
        frame_emotions = bundle.frame_emotions or []
        if not frame_emotions:
            return None, bundle
        fps = getattr(bundle.diagnostics, "sampling_fps", None) or 1.0
        dist = temporal_weighted_average(
            frame_emotions, half_life_seconds=2.0, fps=fps
        )
        return (dist, 1.0), bundle

    def _audio_branch(self, video_path: Path):
        speech = self._audio.analyze(video_path)
        transcript = speech.transcript
        if not transcript:
            return None

        if self._openrouter_key:
            try:
                emo = self._openrouter_emotion(transcript)
                dist = {
                    e: float(emo.get("emotion_scores", {}).get(e, 0.0))
                    for e in EMOTION_LABELS
                }
                return dist, float(emo.get("confidence", 0.5))
            except Exception:
                pass  # fall through to RoBERTa

        txt_sent = self._text.analyze(transcript)
        return caption_to_emotion(txt_sent), float(txt_sent.confidence)

    # ------------------------------------------------------------------
    # Main entry — mirrors the Qwen pipeline's output format
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
            post_id="temporal-post",
            text=text,
            image_path=Path(image_path) if image_path else None,
            video_path=Path(video_path) if video_path else None,
        )

        warnings = []
        video_bundle = None
        transcript = None
        frames_processed = 0

        # caption
        caption = None
        if post.text:
            try:
                caption = self._caption_branch(post.text)
            except Exception as exc:
                warnings.append(f"caption analysis failed: {exc}")

        # video + audio
        video_mod = None
        if post.video_path:
            try:
                video_mod, video_bundle = self._video_branch(post.video_path)
                frames_processed = video_bundle.diagnostics.frames_extracted
                transcript = video_bundle.transcript
                warnings.extend(video_bundle.warnings)
            except Exception as exc:
                warnings.append(f"video analysis failed: {exc}")

        audio_mod = None
        if post.video_path:
            try:
                audio_mod = self._audio_branch(post.video_path)
            except Exception as exc:
                warnings.append(f"audio analysis failed: {exc}")

        # fusion of 8-emotion distributions
        fused_dist, fused_confidence = fuse_emotion_distributions(
            {"text": caption, "visual": video_mod, "speech": audio_mod}
        )

        primary = max(fused_dist, key=fused_dist.get)
        ranked = sorted(fused_dist.items(), key=lambda kv: kv[1], reverse=True)
        secondary = ranked[1][0] if len(ranked) > 1 else None
        valence = valence_of(fused_dist)
        primary_score = round(fused_dist[primary] * fused_confidence, 3)

        emotion_analysis = {
            "emotion_model": "temporal_emotion_net + caption_heuristic + openrouter",
            "primary_emotion": primary,
            "secondary_emotion": secondary,
            "emotion_scores": {k: round(v, 3) for k, v in fused_dist.items()},
            "emotion_distribution": {k: round(v, 3) for k, v in fused_dist.items()},
            "valence": valence,
            "confidence": round(fused_confidence, 3),
            "primary_emotion_score": primary_score,
            "rationale": (
                "Temporal-weighted per-frame emotion net fused with caption "
                "and audio-transcript emotion (confidence-weighted late fusion)."
            ),
        }

        understanding = {
            "content": post.text or (transcript or "No caption provided."),
            "tone": primary,
            "intent": "informational",
            "entities": [],
            "visual_description": (
                f"{frames_processed} video frames processed"
                if post.video_path
                else "No video"
            ),
            "audio_description": (
                "Audio transcribed: " + (transcript[:200] if transcript else "none")
                if post.video_path
                else "No audio"
            ),
            "potential_harm": [],
        }

        result = {
            "execution_time_seconds": round(time.perf_counter() - t0, 3),
            "post_id": post.post_id,
            "understanding": understanding,
            "risk": {"score": 0.0, "labels": [], "probabilities": {}},
            "action": "ALLOW",
            "emotion_analysis": emotion_analysis,
        }
        if warnings:
            result["warnings"] = warnings
        return result
