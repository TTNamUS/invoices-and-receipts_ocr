"""End-to-end orchestrator: image -> OCR -> LLM -> structured extraction.

Shared by both the Streamlit demo and the evaluation runner so the OCR->LLM
flow lives in exactly one place.
"""

from __future__ import annotations

from typing import Optional

from invoice_extractor.llm import LLMExtractor
from invoice_extractor.logging_config import get_logger
from invoice_extractor.ocr import ImageInput, run_ocr
from invoice_extractor.schemas import validate_extraction

logger = get_logger(__name__)


def run_pipeline(
    image: ImageInput,
    *,
    engine: str = "paddleocr",
    model: Optional[str] = None,
    file_name: str = "unknown",
    extractor: Optional[LLMExtractor] = None,
) -> dict:
    """Run OCR then LLM extraction on a single image.

    Returns ``{"ocr": <ocr_dict>, "extraction": <dict>}``. The extraction is
    additionally validated against the Pydantic schemas (best-effort, non-fatal)
    — validation never alters the returned values.

    Pass a shared ``extractor`` to reuse one ``LLMExtractor`` across many calls
    (e.g. the batch runner); otherwise one is created from ``model``/settings.
    """
    logger.info("Pipeline start | file=%s engine=%s", file_name, engine)
    ocr_result = run_ocr(image, engine_name=engine, file_name=file_name)

    if extractor is None:
        extractor = LLMExtractor(model=model)
    extraction = extractor.extract(ocr_result["raw_text"])

    model_obj = validate_extraction(extraction)
    if model_obj is None:
        logger.warning("Extraction for %s did not match a known schema", file_name)

    return {"ocr": ocr_result, "extraction": extraction}
