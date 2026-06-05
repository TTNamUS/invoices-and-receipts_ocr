"""OCR package — engine registry and the ``run_ocr`` entrypoint."""

from __future__ import annotations

from invoice_extractor.ocr.base import ImageInput, OCREngine
from invoice_extractor.ocr.paddle_engine import PaddleOCREngine
from invoice_extractor.ocr.tesseract_engine import TesseractOCREngine

# Lazily-instantiated singletons (PaddleOCR init is expensive).
_engines: dict[str, OCREngine] = {}


def get_engine(engine_name: str = "paddleocr") -> OCREngine:
    """Return a cached OCR engine instance by name."""
    name = "tesseract" if engine_name == "tesseract" else "paddleocr"
    if name not in _engines:
        _engines[name] = TesseractOCREngine() if name == "tesseract" else PaddleOCREngine()
    return _engines[name]


def run_ocr(
    image_input: ImageInput,
    engine_name: str = "paddleocr",
    file_name: str = "unknown",
) -> dict:
    """Top-level OCR function. Returns the standard OCR result dict."""
    return get_engine(engine_name).run(image_input, file_name=file_name)


__all__ = [
    "ImageInput",
    "OCREngine",
    "PaddleOCREngine",
    "TesseractOCREngine",
    "get_engine",
    "run_ocr",
]
