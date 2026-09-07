"""OCR extraction via Tesseract (pytesseract), with graceful degradation."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OcrExtraction:
    """Raw OCR outcome (no sentiment)."""

    available: bool
    text: Optional[str]
    warning: Optional[str] = None


def normalize_ocr_text(raw: str) -> str:
    """Conservative normalization: collapse whitespace, strip noise lines."""
    text = raw.replace("\x0c", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines).strip()


def is_meaningful_ocr_text(text: str, *, min_alnum: int = 3) -> bool:
    alnum = sum(1 for ch in text if ch.isalnum())
    return alnum >= min_alnum


class OcrExtractor:
    """Extract text from images using Tesseract when installed."""

    def __init__(self, lang: str = "eng") -> None:
        self.lang = lang
        self._availability_checked = False
        self._tesseract_available: Optional[bool] = None
        self._unavailable_reason: Optional[str] = None

    def check_availability(self) -> bool:
        """Probe Tesseract once; cache the result."""
        if self._availability_checked:
            return bool(self._tesseract_available)

        self._availability_checked = True
        try:
            import pytesseract
            from pytesseract import TesseractNotFoundError
        except ImportError:
            self._tesseract_available = False
            self._unavailable_reason = "pytesseract package is not installed"
            return False

        try:
            version = pytesseract.get_tesseract_version()
            self._tesseract_available = True
            self._unavailable_reason = None
            logger.info("Tesseract available (version=%s)", version)
            return True
        except TesseractNotFoundError:
            self._tesseract_available = False
            self._unavailable_reason = (
                "OCR unavailable: Tesseract executable not found. "
                "Install Tesseract OCR and ensure it is on PATH "
                "(Windows: https://github.com/UB-Mannheim/tesseract/wiki)."
            )
            logger.warning("%s", self._unavailable_reason)
            return False
        except Exception as exc:  # noqa: BLE001
            self._tesseract_available = False
            self._unavailable_reason = f"OCR unavailable: {exc}"
            logger.warning("%s", self._unavailable_reason)
            return False

    def extract(self, image: Image.Image) -> OcrExtraction:
        """Run OCR; never raises solely because Tesseract is missing."""
        if not self.check_availability():
            return OcrExtraction(
                available=False,
                text=None,
                warning=self._unavailable_reason or "OCR unavailable",
            )

        import pytesseract

        try:
            raw = pytesseract.image_to_string(image, lang=self.lang)
        except Exception as exc:  # noqa: BLE001
            warning = f"OCR failed during extraction: {exc}"
            logger.warning("%s", warning)
            return OcrExtraction(available=False, text=None, warning=warning)

        normalized = normalize_ocr_text(raw)
        if not normalized:
            return OcrExtraction(
                available=True,
                text=None,
                warning="OCR returned no text",
            )
        return OcrExtraction(available=True, text=normalized, warning=None)
