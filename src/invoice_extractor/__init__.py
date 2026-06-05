"""Intelligent Invoice & Receipt Extraction.

End-to-end document AI pipeline: OCR -> LLM -> structured JSON, with evaluation
and a Streamlit demo.
"""

from __future__ import annotations

__version__ = "1.0.0"

from invoice_extractor.config import settings
from invoice_extractor.llm import LLMExtractor
from invoice_extractor.ocr import run_ocr
from invoice_extractor.pipeline import run_pipeline

__all__ = [
    "__version__",
    "settings",
    "run_pipeline",
    "run_ocr",
    "LLMExtractor",
]
