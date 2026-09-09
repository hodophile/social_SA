"""
TemporalEmotionPipeline — Melisa-based pipeline with temporal 8-emotion output.

New architecture (v2):
  1. Extract per-frame SigLIP emotions + OCR text + timestamps
  2. Extract ASR transcript with timed segments
  3. Build a unified narrative timeline
  4. Send the FULL timeline to OpenRouter LLM for synthesis
  5. LLM reasons about temporal dynamics, cross-modal context, sarcasm

Fallback (no OpenRouter key):
  → confidence-weighted mathematical fusion (old behavior)

Output format matches the Qwen2.5-VLM-3B pipeline JSON.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

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
from melisa_temporal_emotion.models import EmotionNet
from melisa_temporal_emotion.fusion import (
    EMOTION_LABELS,
    fuse_emotion_distributions,
    temporal_weighted_average,
    valence_of,
)
from melisa_temporal_emotion.diarization import (
    diarize_segments,
    format_diarization_for_llm,
    add_speakers_to_timeline,
    DiarizationResult,
)


PathLike = Union[str, Path]


@dataclass
class SocialMediaPost:
    """Minimal input record (melisa_poc has no such schema)."""

    post_id: str
    text: Optional[str] = None
    image_path: Optional[Path] = None
    video_path: Optional[Path] = None


@dataclass
class TimelineEvent:
    """Single event on the unified timeline."""

    time_start: float
    time_end: float
    event_type: str  # 'frame', 'ocr', 'asr', 'caption'
    data: dict = field(default_factory=dict)


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
        self._emotion_net = EmotionNet(device="cpu")
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

    # ==================================================================
    # 1. MODALITY BRANCHES (rich outputs, not just distributions)
    # ==================================================================

    def _video_branch(self, video_path: Path) -> Tuple[Optional[dict], Optional[VideoAnalysisBundle], List[dict]]:
        """Returns (avg_dist, bundle, per_frame_data_with_timestamps)."""
        bundle: VideoAnalysisBundle = self._video.analyze(video_path)
        frame_emotions = bundle.frame_emotions or []
        if not frame_emotions:
            return None, bundle, []

        fps = getattr(bundle.diagnostics, "sampling_fps", None) or 1.0
        avg_dist = temporal_weighted_average(
            frame_emotions, half_life_seconds=2.0, fps=fps
        )

        per_frame = []
        for i, emo in enumerate(frame_emotions):
            ts = round(i / fps, 2)
            primary = max(emo, key=emo.get)
            per_frame.append({
                "timestamp": ts,
                "primary_emotion": primary,
                "emotions": {k: round(v, 3) for k, v in emo.items()},
            })

        return avg_dist, bundle, per_frame

    def _image_branch(self, image_path: Path) -> Tuple[Optional[dict], List[dict]]:
        """Returns (dist, per_frame_data)."""
        from PIL import Image
        import numpy as np
        try:
            img = Image.open(image_path).convert("RGB")
            arr = np.asarray(img)
            dist = self._emotion_net.predict_emotion(arr)
            per_frame = [{
                "timestamp": 0.0,
                "primary_emotion": max(dist, key=dist.get),
                "emotions": {k: round(v, 3) for k, v in dist.items()},
            }]
            return dist, per_frame
        except Exception:
            return None, []

    def _audio_branch(self, video_path: Path) -> Tuple[Optional[dict], Optional[float], List[dict], Optional[str]]:
        """Returns (dist, confidence, segments, transcript)."""
        speech = self._audio.analyze(video_path)
        transcript = speech.transcript
        segments = []
        for seg in speech.segments:
            segments.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
            })

        if not transcript:
            return None, None, segments, None

        if self._openrouter_key:
            try:
                emo = self._openrouter_emotion(transcript)
                dist = {
                    e: float(emo.get("emotion_scores", {}).get(e, 0.0))
                    for e in EMOTION_LABELS
                }
                return dist, float(emo.get("confidence", 0.5)), segments, transcript
            except Exception:
                pass

        txt_sent = self._text.analyze(transcript)
        return caption_to_emotion(txt_sent), float(txt_sent.confidence), segments, transcript

    def _caption_branch(self, text: Optional[str]) -> Tuple[Optional[dict], Optional[float]]:
        if not text:
            return None, None
        cap = self._text.analyze(text)
        return caption_to_emotion(cap), float(cap.confidence)

    # ==================================================================
    # 2. UNIFIED TIMELINE
    # ==================================================================

    @staticmethod
    def _build_timeline(
        per_frame_data: List[dict],
        asr_segments: List[dict],
        ocr_text: Optional[str],
        caption: Optional[str],
    ) -> List[TimelineEvent]:
        """Build a single normalized timeline from all modalities."""
        events: List[TimelineEvent] = []

        for f in per_frame_data:
            events.append(TimelineEvent(
                time_start=f["timestamp"],
                time_end=f["timestamp"] + 1.0,
                event_type="frame",
                data={
                    "primary_emotion": f["primary_emotion"],
                    "emotions": f["emotions"],
                },
            ))

        if ocr_text:
            events.append(TimelineEvent(
                time_start=0.0,
                time_end=per_frame_data[-1]["timestamp"] if per_frame_data else 1.0,
                event_type="ocr",
                data={"text": ocr_text},
            ))

        for seg in asr_segments:
            events.append(TimelineEvent(
                time_start=seg["start"],
                time_end=seg["end"],
                event_type="asr",
                data={"text": seg["text"]},
            ))

        if caption:
            events.append(TimelineEvent(
                time_start=0.0,
                time_end=0.0,
                event_type="caption",
                data={"text": caption},
            ))

        events.sort(key=lambda e: e.time_start)
        return events

    # ==================================================================
    # 3. LLM FUSION (rich timeline prompt)
    # ==================================================================

    def _llm_fusion(
        self,
        timeline: List[TimelineEvent],
        diarization: Optional[DiarizationResult] = None,
    ) -> dict:
        """Send unified timeline + diarization to OpenRouter for emotion synthesis."""
        if not self._openrouter_key:
            raise RuntimeError("OPENROUTER_API_KEY not set")

        prompt = self._build_timeline_prompt(timeline, diarization=diarization)

        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self._openrouter_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._openrouter_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": 512,
            },
            timeout=120,
        )
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"]
        start, end = raw.find("{"), raw.rfind("}") + 1
        parsed = json.loads(raw[start:end])

        return {
            "primary_emotion": parsed.get("primary_emotion", "neutral"),
            "secondary_emotion": parsed.get("secondary_emotion") or None,
            "emotion_scores": {
                e: float(parsed.get("emotion_scores", {}).get(e, 0.0))
                for e in EMOTION_LABELS
            },
            "valence": float(parsed.get("valence", 0.0)),
            "confidence": float(parsed.get("confidence", 0.5)),
            "rationale": parsed.get("rationale", ""),
        }

    def _build_timeline_prompt(
        self,
        timeline: List[TimelineEvent],
        diarization: Optional[DiarizationResult] = None,
    ) -> str:
        """Build a narrative prompt from the unified timeline + diarization."""
        lines = [
            "You are an expert multimodal emotion analyst. Analyze the following video content.",
            "",
            "Below is a unified timeline of all extracted modalities (frames, OCR text, speech with speaker labels, caption).",
            "Each event is tagged with its timestamp and speaker (when applicable). Use temporal context, cross-modal agreement/disagreement,",
            "speaker dynamics (who says what, tone shifts between speakers), and any visible text (OCR) to determine the TRUE overall emotion of the video.",
            "",
            "IMPORTANT: The per-frame emotion scores come from a vision model that may misread context.",
            "Your job is to CORRECT and SYNTHESIZE — not just average. Consider:",
            "  - Temporal dynamics (does emotion shift over time?)",
            "  - Cross-modal agreement (do visual, audio, and text agree?)",
            "  - Speaker dynamics (different speakers may convey different emotions)",
            "  - Sarcasm or irony (text says 'love it' but visual shows anger)",
            "  - OCR text visible in frames (signs, captions, subtitles)",
            "",
        ]

        # Add diarization summary if available
        if diarization and diarization.segments:
            lines.append("--- SPEAKER DIARIZATION ---")
            lines.append(f"Detected {diarization.num_speakers} speaker(s): {', '.join(diarization.speaker_labels)}")
            lines.append("")
            for seg in diarization.segments:
                lines.append(f"[{seg.start:.1f}s-{seg.end:.1f}s] {seg.speaker}: \"{seg.text}\"")
            lines.append("")

        lines.extend([
            "--- UNIFIED TIMELINE ---",
            "",
        ])

        for ev in timeline:
            if ev.event_type == "frame":
                emo = ev.data["emotions"]
                top3 = sorted(emo.items(), key=lambda x: x[1], reverse=True)[:3]
                emo_str = ", ".join(f"{k}={v}" for k, v in top3)
                lines.append(f"[{ev.time_start:.1f}s] FRAME → primary={ev.data['primary_emotion']} | {emo_str}")
            elif ev.event_type == "ocr":
                lines.append(f"[OCR] Visible text in frames: \"{ev.data['text']}\"")
            elif ev.event_type == "asr":
                speaker = ev.data.get("speaker", "UNKNOWN")
                lines.append(f"[{ev.time_start:.1f}s-{ev.time_end:.1f}s] SPEECH ({speaker}) → \"{ev.data['text']}\"")
            elif ev.event_type == "caption":
                lines.append(f"[CAPTION] Post text: \"{ev.data['text']}\"")

        lines.extend([
            "",
            "--- TASK ---",
            "",
            "Based on ALL evidence above (including speaker labels and diarization), provide the overall emotion analysis.",
            "Score each emotion from 0.0 to 1.0 (they need NOT sum to 1.0).",
            "",
            "Return ONLY valid JSON:",
            "{",
            '  "primary_emotion": "<one of: joy, sadness, anger, fear, surprise, disgust, trust, anticipation>",',
            '  "secondary_emotion": "<one of the eight or null>",',
            '  "emotion_scores": {',
        ])
        for e in EMOTION_LABELS:
            lines.append(f'    "{e}": 0.0,')
        lines.extend([
            "  },",
            '  "valence": <-1.0 to +1.0>,',
            '  "confidence": 0.0-1.0,',
            '  "rationale": "Explain your reasoning in 1-2 sentences, citing specific timeline events and speakers."',
            "}",
        ])
        return "\n".join(lines)

    # ==================================================================
    # 4. MAIN ENTRY
    # ==================================================================
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
        per_frame_data: List[dict] = []
        asr_segments: List[dict] = []
        ocr_text: Optional[str] = None

        # --- caption branch ---
        caption_dist, caption_conf = None, None
        if post.text:
            try:
                caption_dist, caption_conf = self._caption_branch(post.text)
            except Exception as exc:
                warnings.append(f"caption analysis failed: {exc}")

        # --- video / image / audio branches ---
        video_avg = None
        image_avg = None
        audio_dist = None
        audio_conf = None

        if post.video_path:
            try:
                video_avg, video_bundle, per_frame_data = self._video_branch(post.video_path)
                frames_processed = video_bundle.diagnostics.frames_extracted
                transcript = video_bundle.transcript
                ocr_text = video_bundle.ocr_text
                warnings.extend(video_bundle.warnings)
            except Exception as exc:
                warnings.append(f"video analysis failed: {exc}")

            try:
                audio_dist, audio_conf, asr_segments, _ = self._audio_branch(post.video_path)
            except Exception as exc:
                warnings.append(f"audio analysis failed: {exc}")

        elif post.image_path:
            try:
                image_avg, per_frame_data = self._image_branch(post.image_path)
                frames_processed = 1
            except Exception as exc:
                warnings.append(f"image analysis failed: {exc}")

        # --- Build unified timeline ---
        timeline = self._build_timeline(
            per_frame_data=per_frame_data,
            asr_segments=asr_segments,
            ocr_text=ocr_text,
            caption=post.text,
        )

        # --- Speaker diarization ---
        diarization = None
        if asr_segments:
            try:
                diarization = diarize_segments(asr_segments)
                timeline = add_speakers_to_timeline(timeline, diarization)
            except Exception as exc:
                warnings.append(f"diarization failed: {exc}")

        # --- Fusion: LLM (timeline + diarization) or math fallback ---
        use_llm = bool(self._openrouter_key) and len(timeline) > 0

        if use_llm:
            try:
                llm_result = self._llm_fusion(timeline, diarization=diarization)
                fused_dist = llm_result["emotion_scores"]
                fused_confidence = llm_result["confidence"]
                primary = llm_result["primary_emotion"]
                secondary = llm_result["secondary_emotion"]
                valence = llm_result["valence"]
                rationale = llm_result["rationale"]
                fusion_method = "openrouter_llm_timeline_diarization"
            except Exception as exc:
                warnings.append(f"LLM fusion failed ({exc}); falling back to math fusion")
                use_llm = False

        if not use_llm:
            # Fallback: mathematical fusion
            visual_mod = video_avg or image_avg
            fused_dist, fused_confidence = fuse_emotion_distributions(
                {
                    "text": (caption_dist, caption_conf or 0.5) if caption_dist is not None else None,
                    "visual": (visual_mod, 1.0) if visual_mod is not None else None,
                    "speech": (audio_dist, audio_conf or 0.5) if audio_dist is not None else None,
                }
            )
            primary = max(fused_dist, key=fused_dist.get)
            ranked = sorted(fused_dist.items(), key=lambda kv: kv[1], reverse=True)
            secondary = ranked[1][0] if len(ranked) > 1 else None
            valence = valence_of(fused_dist)
            rationale = (
                "Temporal-weighted per-frame emotion net fused with caption "
                "and audio-transcript emotion (confidence-weighted late fusion)."
            )
            fusion_method = "math_weighted_fusion"

        primary_score = round(fused_dist[primary] * fused_confidence, 3)

        emotion_analysis = {
            "emotion_model": fusion_method,
            "primary_emotion": primary,
            "secondary_emotion": secondary,
            "emotion_scores": {k: round(v, 3) for k, v in fused_dist.items()},
            "emotion_distribution": {k: round(v, 3) for k, v in fused_dist.items()},
            "valence": valence,
            "confidence": round(fused_confidence, 3),
            "primary_emotion_score": primary_score,
            "rationale": rationale,
        }

        # Build understanding block
        visual_desc = f"{frames_processed} video frames processed" if post.video_path else ("1 image processed" if post.image_path else "No visual")
        audio_desc = f"Audio transcribed: {(transcript[:200] if transcript else 'none')}" if post.video_path else "No audio"

        # Add diarization to understanding
        if diarization and diarization.segments:
            audio_desc += f" | Speakers: {', '.join(diarization.speaker_labels)}"

        understanding = {
            "content": post.text or (transcript or "No caption provided."),
            "tone": primary,
            "intent": "informational",
            "entities": [],
            "visual_description": visual_desc,
            "audio_description": audio_desc,
            "potential_harm": [],
        }

        result = {
            "execution_time_seconds": round(time.perf_counter() - t0, 3),
            "post_id": post.post_id,
            "understanding": understanding,
            "risk": {"score": 0.0, "labels": [], "probabilities": {}},
            "action": "ALLOW",
            "emotion_analysis": emotion_analysis,
            "timeline": [
                {
                    "time_start": e.time_start,
                    "time_end": e.time_end,
                    "type": e.event_type,
                    "data": e.data,
                }
                for e in timeline
            ],
        }

        # Add diarization result if available
        if diarization and diarization.segments:
            result["diarization"] = {
                "num_speakers": diarization.num_speakers,
                "speaker_labels": diarization.speaker_labels,
                "segments": [
                    {
                        "start": s.start,
                        "end": s.end,
                        "text": s.text,
                        "speaker": s.speaker,
                    }
                    for s in diarization.segments
                ],
            }

        if warnings:
            result["warnings"] = warnings
        return result
