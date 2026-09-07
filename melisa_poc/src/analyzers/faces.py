"""OpenCV Haar face detection helpers (CPU-friendly, no extra ML deps)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FaceBox:
    """Axis-aligned face box in pixel coordinates (x, y, w, h)."""

    x: int
    y: int
    w: int
    h: int

    def padded(self, image_size: tuple[int, int], pad_ratio: float = 0.25) -> FaceBox:
        width, height = image_size
        pad_x = int(self.w * pad_ratio)
        pad_y = int(self.h * pad_ratio)
        x0 = max(0, self.x - pad_x)
        y0 = max(0, self.y - pad_y)
        x1 = min(width, self.x + self.w + pad_x)
        y1 = min(height, self.y + self.h + pad_y)
        return FaceBox(x=x0, y=y0, w=max(1, x1 - x0), h=max(1, y1 - y0))


@lru_cache(maxsize=1)
def _frontal_cascade():
    import cv2

    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        raise RuntimeError(f"Failed to load Haar cascade: {path}")
    return cascade


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    import cv2

    rgb = np.asarray(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def detect_faces(
    image: Image.Image,
    *,
    min_size: tuple[int, int] = (48, 48),
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
) -> list[FaceBox]:
    """Detect frontal faces. Returns an empty list when none are found."""
    import cv2

    bgr = pil_to_bgr(image)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    cascade = _frontal_cascade()
    raw = cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    boxes = [FaceBox(int(x), int(y), int(w), int(h)) for x, y, w, h in raw]
    boxes.sort(key=lambda b: b.w * b.h, reverse=True)
    return boxes


def crop_faces(
    image: Image.Image,
    boxes: Sequence[FaceBox],
    *,
    pad_ratio: float = 0.25,
    max_faces: int = 5,
) -> list[Image.Image]:
    """Crop padded face regions from a PIL image (largest faces first)."""
    crops: list[Image.Image] = []
    for box in list(boxes)[:max_faces]:
        padded = box.padded(image.size, pad_ratio=pad_ratio)
        crop = image.crop(
            (padded.x, padded.y, padded.x + padded.w, padded.y + padded.h),
        )
        if crop.size[0] >= 24 and crop.size[1] >= 24:
            crops.append(crop.convert("RGB"))
    return crops


def sample_video_frames(
    media_path: object,
    *,
    max_frames: int = 24,
    target_fps: float = 1.0,
) -> list[Image.Image]:
    """Sample RGB frames from a video with OpenCV (no FFmpeg required)."""
    import cv2

    path = str(media_path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {path}")

    frames: list[Image.Image] = []
    try:
        native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if native_fps <= 1e-3:
            native_fps = 25.0
        step = max(int(round(native_fps / max(target_fps, 1e-3))), 1)
        index = 0
        while len(frames) < max_frames:
            ok, bgr = cap.read()
            if not ok:
                break
            if index % step == 0:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb))
            index += 1
    finally:
        cap.release()

    if not frames:
        raise ValueError(f"No frames could be read from video: {path}")
    logger.info("Sampled %s video frames from %s", len(frames), path)
    return frames
