"""
llm_fusion_pipeline.py — Send per-frame SigLIP emotions + timestamps +
audio transcript + caption to OpenRouter for final 8-emotion synthesis.

This is the "timestamp-LLM-fusion" architecture:
    video → frames + timestamps → SigLIP emotions
    audio → Faster-Whisper → transcript
    text → caption
         |
         v    all context in one prompt
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
# Option 1: Set via environment variable (recommended for HF Spaces)
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

# Option 2: Hardcode placeholder (for local testing only — replace with real key)
# OPENROUTER_API_KEY = "sk-or-v1-your-openrouter-key-here"

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


class TimestampLLMFusionPipeline:
    """Pipeline that fuses per-frame emotions + audio + text via LLM."""

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

        # Debug: log key status (never log the actual key)
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
    ) -> str:
        lines = [
            "You are an expert multimodal emotion analyst.",
            "You receive three sources of evidence about a social media post:",
            "  1. Per-frame visual emotion analysis (SigLIP zero-shot)",
            "  2. Audio transcript ( Whisper ASR )",
            "  3. Text caption",
            "",
            "Your task: synthesize ALL evidence into a single coherent 8-emotion assessment.",
            "",
            "=== PER-FRAME VISUAL EVIDENCE ===",
        ]
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
            "Based on ALL the evidence above, produce a JSON object with exactly this shape:",
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
            '  "rationale": "<one sentence explaining how you combined visual, audio, and text evidence>"',
            "}",
            "",
            "Rules:",
            "- emotion_scores must sum to 1.0 (use softmax if needed).",
            "- Consider temporal dynamics: does emotion shift across frames?",
            "- Audio transcript can override visual if they strongly disagree.",
            "- Caption text provides intent/context that visual alone may miss.",
            "- Return ONLY the JSON object, no markdown, no explanation outside JSON.",
        ])
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Call Kimi (Moonshot AI)
    # ------------------------------------------------------------------
    def _call_llm(self, prompt: str) -> Dict[str, Any]:
        if not self._api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set")

        # Log prompt
        print("\n" + "="*80)
        print("PROMPT SENT TO OPENROUTER")
        print("="*80)
        print(prompt)
        print("="*80 + "\n")

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

        # Provide clearer errors for common HTTP failures
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
        print("\n" + "="*80)
        print("RAW RESPONSE FROM OPENROUTER")
        print("="*80)
        print(raw)
        print("="*80 + "\n")

        # Extract JSON
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

        # 2. Audio: Whisper transcript
        transcript: Optional[str] = None
        if video_path:
            try:
                speech = self._audio.analyze(video_path)
                transcript = speech.transcript
            except Exception as exc:
                warnings.append(f"Audio analysis failed: {exc}")

        # 3. LLM fusion
        emotion_analysis: Dict[str, Any] = {}
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
            llm_debug = None
            try:
                prompt = self._build_prompt(frame_results, transcript, text)
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
            # Text-only: use simple heuristic
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

        # Average frame emotions
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
