"""
per_frame_siglip.py — Extract frames from video/image with timestamps,
run SigLIP zero-shot emotion classification on each frame.

Output: list of {timestamp_seconds, emotion_scores} dicts.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# melisa_poc internal modules import each other as `src.*`
_MELISA_ROOT = Path(__file__).resolve().parent / "melisa_poc"
if str(_MELISA_ROOT) not in sys.path:
    sys.path.insert(0, str(_MELISA_ROOT))

warnings.filterwarnings("ignore", message=".*pretrained.*deprecated.*")
warnings.filterwarnings("ignore", message=".*Arguments other than.*deprecated.*")

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


class SigLIPFrameAnalyzer:
    """Extract frames with timestamps and classify emotions via SigLIP."""

    _MODEL = None
    _PROCESSOR = None
    _DEVICE = None

    def __init__(self, device: str = "cpu"):
        self.device = device

    def _load(self):
        if SigLIPFrameAnalyzer._MODEL is not None:
            return
        from transformers import AutoProcessor, AutoModel
        model_name = "google/siglip-base-patch16-224"
        SigLIPFrameAnalyzer._PROCESSOR = AutoProcessor.from_pretrained(model_name)
        SigLIPFrameAnalyzer._MODEL = AutoModel.from_pretrained(model_name)
        SigLIPFrameAnalyzer._DEVICE = self.device
        SigLIPFrameAnalyzer._MODEL.to(self.device).eval()

    @torch.no_grad()
    def _classify_frame(self, pil_img: Image.Image) -> Dict[str, float]:
        self._load()
        image_inputs = SigLIPFrameAnalyzer._PROCESSOR(
            images=pil_img, return_tensors="pt",
        ).to(SigLIPFrameAnalyzer._DEVICE)
        text_inputs = SigLIPFrameAnalyzer._PROCESSOR(
            text=list(EMOTION_PROMPTS.values()),
            return_tensors="pt", padding=True,
        ).to(SigLIPFrameAnalyzer._DEVICE)
        outputs = SigLIPFrameAnalyzer._MODEL(**image_inputs, **text_inputs)
        probs = F.softmax(outputs.logits_per_image[0], dim=0).cpu().numpy()
        return {
            emotion: round(float(probs[i]), 4)
            for i, emotion in enumerate(EMOTION_PROMPTS.keys())
        }

    def analyze_video(
        self,
        video_path: Path,
        sample_fps: float = 1.0,
        max_frames: int = 20,
    ) -> List[Dict]:
        """Extract frames at sample_fps, classify each with SigLIP.
        Returns [{timestamp_seconds, emotions: {joy: 0.3, ...}}, ...]
        """
        import tempfile
        import shutil
        from melisa_poc.src.media.ffmpeg_utils import probe_video
        from melisa_poc.src.media.samplers import build_frame_sampler, VideoSamplingConfig
        from melisa_poc.src.analyzers.visual import VisualSentimentAnalyzer

        video_path = Path(video_path)
        probe = probe_video(video_path)
        duration = probe.duration_seconds

        # Build frame sampler with the requested fps
        sampling = VideoSamplingConfig(fps=sample_fps)
        sampler = build_frame_sampler("fixed_fps", sampling=sampling)

        # Extract frames to temp directory
        tmpdir = tempfile.mkdtemp(prefix="siglip_frames_")
        try:
            sampled = sampler.sample(
                media_path=video_path,
                output_dir=tmpdir,
                duration_seconds=duration,
            )

            frames = sampled.frames
            # Cap frames
            if len(frames) > max_frames:
                step = len(frames) // max_frames
                frames = frames[::step][:max_frames]

            results = []
            for frame_info in frames:
                try:
                    pil_img = VisualSentimentAnalyzer.load_image(frame_info.path)
                    emotions = self._classify_frame(pil_img)
                    results.append({
                        "timestamp_seconds": round(frame_info.timestamp_seconds, 2),
                        "emotions": emotions,
                    })
                except Exception as exc:
                    results.append({
                        "timestamp_seconds": round(frame_info.timestamp_seconds, 2),
                        "emotions": None,
                        "error": str(exc),
                    })
            return results
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def analyze_image(self, image_path: Path) -> List[Dict]:
        """Treat single image as a 1-frame video at t=0."""
        from melisa_poc.src.analyzers.visual import VisualSentimentAnalyzer
        pil_img = VisualSentimentAnalyzer.load_image(image_path)
        emotions = self._classify_frame(pil_img)
        return [{"timestamp_seconds": 0.0, "emotions": emotions}]
