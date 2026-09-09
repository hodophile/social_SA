"""
llm_fusion_pipeline.py — Send per-frame SigLIP emotions + timestamps +
audio transcript (with speaker diarization) + caption to OpenRouter for
final 8-emotion synthesis.

Architecture:
    video → frames + timestamps → SigLIP emotions
    audio → Faster-Whisper → transcript + segments
            → diarization (pause-based speaker segmentation)
    text → caption
         |
         v    all context in one prompt (with speaker labels)
    OpenRouter (GPT-4o-mini)
         |
         v    structured JSON
    8-emotion distribution + valence + confidence
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

# melisa_poc internal modules import each other as `src.*`
_MELISA_ROOT = Path(__file__).resolve().parent / "melisa_poc"
if str(_MELISA_ROOT) not in sys.path:
    sys.path.insert(0, str(_MELISA_ROOT))

from per_frame_siglip import SigLIPFrameAnalyzer
from melisa_poc.src.analyzers.audio import AudioAnalyzer

# ------------------------------------------------------------------
# OpenRouter API configuration
# ------------------------------------------------------------------
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

EMOTION_LABELS = [
    "joy", "sadness", "anger", "fear",
    "surprise", "disgust", "trust", "anticipation",
]

VALENCE_MAP = {
    "joy": 1.0, "trust": 0.5, "anticipation": 0.3,
    "surprise": 0.0, "fear": -0.5, "anger": -0.8,
    "sadness": -1.0, "disgust": -0.8,
}

# ------------------------------------------------------------------
# Logging helpers
# ------------------------------------------------------------------
_LOG_DIR = Path("/tmp")
_LOG_FILE = _LOG_DIR / "openrouter_prompts.log"


def _log_to_file(label: str, content: str) -> None:
    """Append labeled content to the shared log file."""
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(_LOG_FILE, "a", encoding="utf-8") as fh:
            fh.write(f"\n{'=' * 80}\n")
            fh.write(f"[{label}] {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            fh.write(f"{'=' * 80}\n")
            fh.write(content)
            fh.write("\n")
    except Exception as exc:
        print(f"[LOG ERROR] Could not write to {_LOG_FILE}: {exc}")


# ------------------------------------------------------------------
# Speaker diarization (lightweight, pause-based)
# ------------------------------------------------------------------
def _diarize_segments(segments: List[Dict[str, Any]], pause_threshold: float = 1.0) -> List[Dict[str, Any]]:
    """Label Whisper segments with speaker IDs based on pause gaps."""
    if not segments:
        return []

    sorted_segs = sorted(segments, key=lambda s: s.get("start", 0))
    diarized: List[Dict[str, Any]] = []
    speaker_idx = 0

    for i, seg in enumerate(sorted_segs):
        if i > 0:
            gap = seg.get("start", 0) - sorted_segs[i - 1].get("end", 0)
            if gap > pause_threshold:
                speaker_idx += 1
        diarized.append({
            "start": seg.get("start", 0),
            "end": seg.get("end", 0),
            "text": seg.get("text", ""),
            "speaker": f"Speaker {chr(ord('A') + speaker_idx)}",
        })

    # Merge consecutive same-speaker segments
    merged: List[Dict[str, Any]] = []
    for seg in diarized:
        if merged and merged[-1]["speaker"] == seg["speaker"]:
            merged[-1]["end"] = seg["end"]
            merged[-1]["text"] = (merged[-1]["text"] + " " + seg["text"]).strip()
        else:
            merged.append(seg.copy())
    return merged


class TimestampLLMFusionPipeline:
    """Pipeline that fuses per-frame emotions + audio (with diarization) + text via LLM."""

    def __init__(
        self,
        siglip_analyzer: Optional[SigLIPFrameAnalyzer] = None,
        audio_analyzer: Optional[AudioAnalyzer] = None,
        openrouter_api_key: Optional[str] = None,
        openrouter_model: str = "openai/gpt-4o-mini",
    ):
        self._siglip = siglip_analyzer or SigLIPFrameAnalyzer(device="cpu")
        self._audio = audio_analyzer or AudioAnalyzer(
            whisper_model="base.en",
            compute_type="int8",
            language="en",
        )
        self._api_key = openrouter_api_key or OPENROUTER_API_KEY
        self._model = openrouter_model or OPENROUTER_MODEL

        import logging
        logger = logging.getLogger(__name__)
        if self._api_key:
            masked = self._api_key[:8] + "..." + self._api_key[-4:] if len(self._api_key) > 12 else "***"
            logger.info("TimestampLLMFusionPipeline: API key detected (%s)", masked)
        else:
            logger.warning("TimestampLLMFusionPipeline: NO API KEY found. Set OPENROUTER_API_KEY env var.")

    # ------------------------------------------------------------------
    # Build the prompt
    # ------------------------------------------------------------------
    def _build_prompt(
        self,
        frame_results: List[Dict],
        transcript: Optional[str],
        caption: Optional[str],
        diarized_segments: Optional[List[Dict]] = None,
    ) -> str:
        lines = [
            "You are an expert multimodal emotion analyst.",
            "You receive multiple sources of evidence about a social media post:",
            "  1. Per-frame visual emotion analysis (SigLIP zero-shot)",
            "  2. Audio transcript with speaker diarization (Whisper ASR + pause-based segmentation)",
            "  3. Text caption",
            "",
            "Your task: synthesize ALL evidence into a single coherent 8-emotion assessment.",
            "",
        ]

        # Diarization section
        if diarized_segments:
            unique_speakers = sorted({s["speaker"] for s in diarized_segments})
            lines.append("=== SPEAKER DIARIZATION ===")
            lines.append(f"Detected {len(unique_speakers)} speaker(s): {', '.join(unique_speakers)}")
            lines.append("")
            for seg in diarized_segments:
                lines.append(f"  [{seg['start']:.1f}s-{seg['end']:.1f}s] {seg['speaker']}: \"{seg['text']}\"")
            lines.append("")

        # Frame evidence
        lines.append("=== PER-FRAME VISUAL EVIDENCE ===")
        for fr in frame_results:
            ts = fr.get("timestamp_seconds", 0)
            emotions = fr.get("emotions")
            if emotions:
                emo_str = ", ".join(f"{k}={v:.3f}" for k, v in emotions.items())
                lines.append(f"  t={ts:5.2f}s: {emo_str}")
            else:
                err = fr.get("error", "unknown error")
                lines.append(f"  t={ts:5.2f}s: [failed: {err}]")

        lines.extend([
            "",
            "=== AUDIO TRANSCRIPT ===",
            f"  {transcript or 'No audio transcript available.'}",
            "",
            "=== TEXT CAPTION ===",
            f"  {caption or 'No caption provided.'}",
            "",
            "=== INSTRUCTIONS ===",
            "Based on ALL the evidence above (including speaker labels), produce a JSON object with exactly this shape:",
            "{",
            '  "primary_emotion": "<one of: joy, sadness, anger, fear, surprise, disgust, trust, anticipation>",',
            '  "secondary_emotion": "<one of the eight or null>",',
            '  "emotion_scores": {',
        ])
        for e in EMOTION_LABELS:
            lines.append(f'    "{e}": 0.000,')
        lines.extend([
            "  },",
            '  "valence": -1.0 to +1.0,',
            '  "confidence": 0.0 to 1.0,',
            '  "rationale": "<one sentence explaining how you combined visual, audio, and speaker evidence>"',
            "}",
            "",
            "Rules:",
            "- emotion_scores must sum to 1.0 (use softmax if needed).",
            "- Consider temporal dynamics: does emotion shift across frames?",
            "- Speaker dynamics: different speakers may convey different emotions.",
            "- Audio transcript can override visual if they strongly disagree.",
            "- Caption text provides intent/context that visual alone may miss.",
            "- Return ONLY the JSON object, no markdown, no explanation outside JSON.",
        ])
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Call OpenRouter
    # ------------------------------------------------------------------
    def _call_llm(self, prompt: str) -> Dict[str, Any]:
        if not self._api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set")

        # Log to stdout and file
        print("\n" + "=" * 80)
        print("PROMPT SENT TO OPENROUTER")
        print("=" * 80)
        print(prompt)
        print("=" * 80 + "\n")
        _log_to_file("PROMPT", prompt)

        r = requests.post(
            OPENROUTER_API_URL,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://huggingface.co/spaces/wsm26/sa_smp_llm_fusion",
                "X-Title": "SA Social Media Sentiment Analysis",
            },
            json={
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": 512,
            },
            timeout=120,
        )

        if r.status_code == 401:
            raise RuntimeError("OpenRouter returned 401 Unauthorized — check your API key")
        if r.status_code == 404:
            raise RuntimeError(f"OpenRouter returned 404 — model '{self._model}' may not exist")
        if r.status_code == 429:
            raise RuntimeError("OpenRouter rate limit exceeded — try again later")
        if r.status_code >= 500:
            raise RuntimeError(f"OpenRouter server error {r.status_code} — try again later")

        r.raise_for_status()
        resp = r.json()

        if "choices" not in resp or not resp["choices"]:
            raise RuntimeError(f"OpenRouter API returned empty choices: {json.dumps(resp)[:200]}")

        raw = resp["choices"][0]["message"]["content"]

        # Log response
        print("\n" + "=" * 80)
        print("RAW RESPONSE FROM OPENROUTER")
        print("=" * 80)
        print(raw)
        print("=" * 80 + "\n")
        _log_to_file("RESPONSE", raw)

        start, end = raw.find("{"), raw.rfind("}") + 1
        if start == -1 or end <= start:
            raise ValueError(f"LLM did not return JSON: {raw[:200]}")
        parsed = json.loads(raw[start:end])
        parsed["_debug"] = {"prompt_sent": prompt, "raw_response": raw}
        return parsed

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def analyze(
        self,
        text: Optional[str] = None,
        image_path: Optional[Path] = None,
        video_path: Optional[Path] = None,
        sample_fps: float = 1.0,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        warnings: List[str] = []

        # 1. Visual: per-frame SigLIP emotions with timestamps
        frame_results: List[Dict] = []
        if video_path:
            try:
                frame_results = self._siglip.analyze_video(
                    Path(video_path), sample_fps=sample_fps
                )
            except Exception as exc:
                warnings.append(f"SigLIP video analysis failed: {exc}")
        elif image_path:
            try:
                frame_results = self._siglip.analyze_image(Path(image_path))
            except Exception as exc:
                warnings.append(f"SigLIP image analysis failed: {exc}")

        # 2. Audio: Whisper transcript + segments for diarization
        transcript: Optional[str] = None
        asr_segments: List[Dict] = []
        if video_path:
            try:
                speech = self._audio.analyze(video_path)
                transcript = speech.transcript
                # Extract segments for diarization
                for seg in speech.segments:
                    asr_segments.append({
                        "start": seg.start,
                        "end": seg.end,
                        "text": seg.text,
                    })
            except Exception as exc:
                warnings.append(f"Audio analysis failed: {exc}")

        # 3. Speaker diarization
        diarized_segments = _diarize_segments(asr_segments) if asr_segments else None

        # 4. LLM fusion
        emotion_analysis: Dict[str, Any] = {}
        llm_debug = None
        if not self._api_key:
            warnings.append("OPENROUTER_API_KEY not set — set it in Space Settings → Secrets")
            emotion_analysis = self._fallback_emotion(
                frame_results, text, reason="OPENROUTER_API_KEY not configured"
            )
        elif not frame_results:
            warnings.append("No visual frames available for LLM fusion")
            emotion_analysis = self._fallback_emotion(
                frame_results, text, reason="No visual frames to analyze"
            )
        else:
            try:
                prompt = self._build_prompt(
                    frame_results, transcript, text,
                    diarized_segments=diarized_segments,
                )
                llm_out = self._call_llm(prompt)
                emotion_analysis = {
                    "emotion_model": f"timestamp-llm-fusion ({self._model})",
                    "primary_emotion": llm_out.get("primary_emotion"),
                    "secondary_emotion": llm_out.get("secondary_emotion"),
                    "emotion_scores": llm_out.get("emotion_scores", {}),
                    "emotion_distribution": llm_out.get("emotion_scores", {}),
                    "valence": llm_out.get("valence", 0.0),
                    "confidence": llm_out.get("confidence", 0.0),
                    "primary_emotion_score": round(
                        llm_out.get("emotion_scores", {}).get(
                            llm_out.get("primary_emotion"), 0.0
                        ) * llm_out.get("confidence", 0.0), 3
                    ),
                    "rationale": llm_out.get("rationale", ""),
                }
                llm_debug = llm_out.get("_debug")
            except Exception as exc:
                err_msg = f"{type(exc).__name__}: {exc}"
                warnings.append(f"LLM fusion failed: {err_msg}")
                emotion_analysis = self._fallback_emotion(
                    frame_results, text, reason=f"OpenRouter API error: {err_msg}"
                )

        # Build understanding
        primary = emotion_analysis.get("primary_emotion", "trust")
        understanding = {
            "content": text or (transcript or "No caption provided."),
            "tone": primary,
            "intent": "informational",
            "entities": [],
            "visual_description": (
                f"{len(frame_results)} frames analyzed with timestamps"
                if frame_results else "No visual"
            ),
            "audio_description": (
                "Audio transcribed: " + (transcript[:200] if transcript else "none")
                if video_path else "No audio"
            ),
            "potential_harm": [],
        }

        result = {
            "execution_time_seconds": round(time.perf_counter() - t0, 3),
            "post_id": "timestamp-llm-post",
            "understanding": understanding,
            "risk": {"score": 0.0, "labels": [], "probabilities": {}},
            "action": "ALLOW",
            "emotion_analysis": emotion_analysis,
            "frame_timestamps": frame_results,
        }

        # Add diarization
        if diarized_segments:
            result["diarization"] = {
                "num_speakers": len({s["speaker"] for s in diarized_segments}),
                "speaker_labels": sorted({s["speaker"] for s in diarized_segments}),
                "segments": diarized_segments,
            }

        # Add LLM debug
        if llm_debug:
            result["_llm_debug"] = llm_debug

        if warnings:
            result["warnings"] = warnings
        return result

    def _fallback_emotion(
        self,
        frame_results: List[Dict],
        caption: Optional[str],
        reason: str = "LLM not configured",
    ) -> Dict[str, Any]:
        """Heuristic fallback when LLM is unavailable."""
        if not frame_results:
            if caption and ("happy" in caption.lower() or "love" in caption.lower() or "excited" in caption.lower()):
                primary = "joy"
            elif caption and ("sad" in caption.lower() or "cry" in caption.lower()):
                primary = "sadness"
            elif caption and ("angry" in caption.lower() or "hate" in caption.lower()):
                primary = "anger"
            else:
                primary = "trust"
            scores = {e: 0.0 for e in EMOTION_LABELS}
            scores[primary] = 1.0
            return {
                "emotion_model": "fallback heuristic (no LLM)",
                "primary_emotion": primary,
                "secondary_emotion": None,
                "emotion_scores": scores,
                "emotion_distribution": scores,
                "valence": VALENCE_MAP.get(primary, 0.0),
                "confidence": 0.3,
                "primary_emotion_score": 0.3,
                "rationale": f"Fallback: {reason}. Used text heuristic.",
            }

        avg = {e: 0.0 for e in EMOTION_LABELS}
        valid = 0
        for fr in frame_results:
            emo = fr.get("emotions")
            if emo:
                valid += 1
                for e, v in emo.items():
                    avg[e] += v
        if valid > 0:
            for e in avg:
                avg[e] /= valid
        total = sum(avg.values())
        if total > 0:
            avg = {e: round(v / total, 3) for e, v in avg.items()}
        primary = max(avg, key=avg.get)
        return {
            "emotion_model": "fallback heuristic (no LLM)",
            "primary_emotion": primary,
            "secondary_emotion": None,
            "emotion_scores": avg,
            "emotion_distribution": avg,
            "valence": round(sum(avg.get(e, 0) * VALENCE_MAP.get(e, 0) for e in EMOTION_LABELS), 3),
            "confidence": 0.5,
            "primary_emotion_score": round(avg[primary] * 0.5, 3),
            "rationale": f"Fallback: {reason}. Averaged per-frame SigLIP scores.",
        }
