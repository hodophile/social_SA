"""Facial-expression sentiment via face crops (or face-gated full frame) + SigLIP."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Union

from PIL import Image

from src.analyzers.faces import crop_faces, detect_faces, sample_video_frames
from src.analyzers.visual import VisualSentimentAnalyzer
from src.config import (
    DEFAULT_FACE_GATE_PROMPTS,
    DEFAULT_FACIAL_EXPRESSION_PROMPTS,
    DEFAULT_VISUAL_MODEL,
    FACE_GATE_MIN_PROBABILITY,
)
from src.fusion import aggregate_frame_visual_scores
from src.schemas import SentimentEvidence

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


@dataclass
class FacialExpressionOutcome:
    """Result of facial-expression analysis for one image or video."""

    faces_detected: int
    face_like: bool
    evidence: Optional[SentimentEvidence] = None
    warnings: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.evidence is not None and self.face_like


class FacialExpressionAnalyzer:
    """Score facial expressions when a face (or face-like content) is present.

    Detection path:
    1. OpenCV Haar frontal-face boxes → crop → SigLIP facial prompts
    2. If no Haar boxes, SigLIP face-gate on the full frame (helps stylized faces)
    3. Otherwise skip — no invented scores
    """

    def __init__(
        self,
        *,
        visual_analyzer: Optional[VisualSentimentAnalyzer] = None,
        face_gate_min_probability: float = FACE_GATE_MIN_PROBABILITY,
        max_faces: int = 5,
        video_max_frames: int = 24,
        video_target_fps: float = 1.0,
    ) -> None:
        self._visual = visual_analyzer or VisualSentimentAnalyzer(
            model_name=DEFAULT_VISUAL_MODEL,
            prompts=DEFAULT_FACIAL_EXPRESSION_PROMPTS,
        )
        self.face_gate_min_probability = face_gate_min_probability
        self.max_faces = max_faces
        self.video_max_frames = video_max_frames
        self.video_target_fps = video_target_fps

    def load(self) -> None:
        self._visual.load()

    def analyze_path(self, media_path: PathLike) -> FacialExpressionOutcome:
        image = VisualSentimentAnalyzer.load_image(media_path)
        return self.analyze_image(image)

    def analyze_image(self, image: Image.Image) -> FacialExpressionOutcome:
        rgb = image.convert("RGB")
        boxes = detect_faces(rgb)
        crops = crop_faces(rgb, boxes, max_faces=self.max_faces)
        warnings: list[str] = []
        details: dict = {
            "haar_faces": len(boxes),
            "crops_scored": 0,
            "method": None,
        }

        if crops:
            scores = [self._score_expression(crop, source="face_crop") for crop in crops]
            details["crops_scored"] = len(scores)
            details["method"] = "haar_face_crop"
            evidence = (
                scores[0]
                if len(scores) == 1
                else aggregate_frame_visual_scores(scores)
            )
            assert evidence is not None
            evidence = evidence.model_copy(
                update={
                    "details": {
                        **(evidence.details or {}),
                        "facial_expression": details,
                        "prompts": dict(DEFAULT_FACIAL_EXPRESSION_PROMPTS),
                    },
                },
            )
            return FacialExpressionOutcome(
                faces_detected=len(boxes),
                face_like=True,
                evidence=evidence,
                warnings=warnings,
                details=details,
            )

        gate = self._face_gate(rgb)
        details["face_gate"] = gate
        if gate.get("face_probability", 0.0) >= self.face_gate_min_probability:
            details["method"] = "face_gated_full_frame"
            evidence = self._score_expression(rgb, source="face_gated_full_frame")
            evidence = evidence.model_copy(
                update={
                    "details": {
                        **(evidence.details or {}),
                        "facial_expression": details,
                        "prompts": dict(DEFAULT_FACIAL_EXPRESSION_PROMPTS),
                    },
                },
            )
            warnings.append(
                "No Haar face box found; scored full frame after face-like content gate.",
            )
            return FacialExpressionOutcome(
                faces_detected=0,
                face_like=True,
                evidence=evidence,
                warnings=warnings,
                details=details,
            )

        details["method"] = "no_face"
        return FacialExpressionOutcome(
            faces_detected=0,
            face_like=False,
            evidence=None,
            warnings=warnings,
            details=details,
        )

    def analyze_video_path(self, media_path: PathLike) -> FacialExpressionOutcome:
        frames = sample_video_frames(
            media_path,
            max_frames=self.video_max_frames,
            target_fps=self.video_target_fps,
        )
        return self.analyze_frames(frames)

    def analyze_frames(self, frames: Sequence[Image.Image]) -> FacialExpressionOutcome:
        frame_scores: list[SentimentEvidence] = []
        total_haar = 0
        gated_frames = 0
        warnings: list[str] = []

        for frame in frames:
            outcome = self.analyze_image(frame)
            total_haar += outcome.faces_detected
            if outcome.ok and outcome.evidence is not None:
                frame_scores.append(outcome.evidence)
                if outcome.details.get("method") == "face_gated_full_frame":
                    gated_frames += 1
            warnings.extend(outcome.warnings)

        details = {
            "frames_sampled": len(frames),
            "frames_with_expression": len(frame_scores),
            "haar_faces_total": total_haar,
            "face_gated_frames": gated_frames,
            "method": "video_frame_aggregate" if frame_scores else "no_face",
        }

        if not frame_scores:
            return FacialExpressionOutcome(
                faces_detected=total_haar,
                face_like=False,
                evidence=None,
                warnings=list(dict.fromkeys(warnings)),
                details=details,
            )

        evidence = (
            frame_scores[0]
            if len(frame_scores) == 1
            else aggregate_frame_visual_scores(frame_scores)
        )
        assert evidence is not None
        evidence = evidence.model_copy(
            update={
                "details": {
                    **(evidence.details or {}),
                    "facial_expression": details,
                    "prompts": dict(DEFAULT_FACIAL_EXPRESSION_PROMPTS),
                },
            },
        )
        return FacialExpressionOutcome(
            faces_detected=total_haar,
            face_like=True,
            evidence=evidence,
            warnings=list(dict.fromkeys(warnings)),
            details=details,
        )

    def _score_with_prompts(
        self,
        image: Image.Image,
        prompts: dict[str, str],
    ) -> SentimentEvidence:
        previous = dict(self._visual.prompts)
        try:
            self._visual.prompts = dict(prompts)
            return self._visual.analyze_image(image)
        finally:
            self._visual.prompts = previous

    def _score_expression(self, image: Image.Image, *, source: str) -> SentimentEvidence:
        evidence = self._score_with_prompts(image, DEFAULT_FACIAL_EXPRESSION_PROMPTS)
        return evidence.model_copy(
            update={
                "details": {
                    **(evidence.details or {}),
                    "source": source,
                    "analysis_kind": "facial_expression",
                },
            },
        )

    def _face_gate(self, image: Image.Image) -> dict:
        """Return face vs no-face probabilities using SigLIP concept prompts."""
        evidence = self._score_with_prompts(image, DEFAULT_FACE_GATE_PROMPTS)
        probs = evidence.probabilities or {}
        face_p = float(probs.get("positive", 0.0))
        no_face_p = float(probs.get("negative", 0.0))
        return {
            "face_probability": face_p,
            "no_face_probability": no_face_p,
            "label": evidence.label,
            "threshold": self.face_gate_min_probability,
        }
