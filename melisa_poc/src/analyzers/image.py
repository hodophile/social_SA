"""Image activity analysis: visual zero-shot + OCR (+ optional caption via pipeline)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from PIL import Image

from src.analyzers.ocr import OcrExtractor, is_meaningful_ocr_text
from src.analyzers.visual import VisualSentimentAnalyzer
from src.analyzers.text import TextSentimentAnalyzer
from src.config import OCR_MIN_ALNUM_CHARS
from src.schemas import SentimentEvidence

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


@dataclass
class ImageModalityEvidence:
    """Separate evidence streams for one image activity (pre-fusion)."""

    visual: SentimentEvidence
    ocr_text: Optional[str] = None
    ocr_sentiment: Optional[SentimentEvidence] = None
    warnings: list[str] = field(default_factory=list)


class ImageAnalyzer:
    """Compose visual scoring and OCR for a local image file."""

    def __init__(
        self,
        visual_analyzer: Optional[VisualSentimentAnalyzer] = None,
        ocr_extractor: Optional[OcrExtractor] = None,
        text_analyzer: Optional[TextSentimentAnalyzer] = None,
    ) -> None:
        self._visual = visual_analyzer or VisualSentimentAnalyzer()
        self._ocr = ocr_extractor or OcrExtractor()
        # Shared text analyzer is injected by the pipeline when available.
        self._text = text_analyzer

    def set_text_analyzer(self, text_analyzer: TextSentimentAnalyzer) -> None:
        self._text = text_analyzer

    def analyze_path(self, media_path: PathLike) -> ImageModalityEvidence:
        image = VisualSentimentAnalyzer.load_image(media_path)
        return self.analyze_image(image)

    def extract_ocr_evidence(
        self,
        image: Image.Image,
    ) -> tuple[Optional[str], Optional[SentimentEvidence], list[str]]:
        """Optional OCR path. Missing Tesseract never fails visual sentiment."""
        warnings: list[str] = []
        ocr = self._ocr.extract(image)

        ocr_text: Optional[str] = None
        ocr_sentiment: Optional[SentimentEvidence] = None

        if not ocr.available:
            if ocr.warning:
                warnings.append(ocr.warning)
        elif ocr.text is None:
            if ocr.warning:
                warnings.append(ocr.warning)
        elif not is_meaningful_ocr_text(ocr.text, min_alnum=OCR_MIN_ALNUM_CHARS):
            warnings.append("OCR text present but not meaningful after normalization")
            ocr_text = ocr.text
        else:
            ocr_text = ocr.text
            if self._text is None:
                warnings.append("OCR text found but text sentiment analyzer is not configured")
            else:
                try:
                    ocr_sentiment = self._text.analyze(ocr_text)
                    # Annotate so OCR evidence is distinguishable from caption text.
                    ocr_sentiment = ocr_sentiment.model_copy(
                        update={
                            "details": {
                                **(ocr_sentiment.details or {}),
                                "source": "ocr",
                                "extracted_text_preview": ocr_text[:200],
                            },
                        },
                    )
                except ValueError as exc:
                    warnings.append(f"OCR text could not be scored: {exc}")
        return ocr_text, ocr_sentiment, warnings

    def analyze_image(self, image: Image.Image) -> ImageModalityEvidence:
        warnings: list[str] = []

        visual = self._visual.analyze_image(image)
        ocr_text, ocr_sentiment, ocr_warnings = self.extract_ocr_evidence(image)
        warnings.extend(ocr_warnings)

        logger.info(
            "Image analysis complete visual=%s ocr_chars=%s warnings=%s",
            visual.label,
            len(ocr_text) if ocr_text else 0,
            len(warnings),
        )
        return ImageModalityEvidence(
            visual=visual,
            ocr_text=ocr_text,
            ocr_sentiment=ocr_sentiment,
            warnings=warnings,
        )
