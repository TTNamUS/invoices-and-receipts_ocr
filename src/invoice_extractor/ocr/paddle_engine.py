"""PaddleOCR engine (primary)."""

from __future__ import annotations

from invoice_extractor.config import settings
from invoice_extractor.logging_config import get_logger
from invoice_extractor.ocr.base import (
    ImageInput,
    OCREngine,
    assemble_result,
    load_image,
    pil_to_bgr,
    polygon_to_bbox,
)

logger = get_logger(__name__)


class PaddleOCREngine(OCREngine):
    name = "paddleocr"

    def __init__(self, lang: str | None = None, use_gpu: bool | None = None):
        self.lang = lang if lang is not None else settings.paddle_lang
        self.use_gpu = use_gpu if use_gpu is not None else settings.paddle_use_gpu
        self._ocr = None
        self._init_error = None  # cache init failure so we don't retry after partial PDX init

    def _get_ocr(self):
        if self._init_error is not None:
            raise self._init_error
        if self._ocr is None:
            from paddleocr import PaddleOCR

            try:
                # PaddleOCR v3 notes:
                # - ocr_version='PP-OCRv4' selects mobile det+rec models for 'en',
                #   avoiding the heavy PP-OCRv5_server_det default which is too
                #   slow for CPU inference without mkldnn.
                # - Disable doc orientation/unwarping sub-models (not needed for
                #   flat invoice/receipt images) to cut load time and memory.
                # - engine_config disables mkldnn run_mode and PIR executor to avoid
                #   "ConvertPirAttribute2RuntimeAttribute not support" on Windows.
                logger.info("Initializing PaddleOCR (lang=%s)...", self.lang)
                self._ocr = PaddleOCR(
                    lang=self.lang,
                    ocr_version="PP-OCRv4",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    engine_config={"run_mode": "paddle", "enable_new_ir": False},
                )
            except Exception as exc:
                self._init_error = exc
                logger.error("PaddleOCR init failed: %s", exc)
                raise
        return self._ocr

    def run(self, image_input: ImageInput, file_name: str = "unknown") -> dict:
        image = load_image(image_input)
        bgr = pil_to_bgr(image)
        ocr = self._get_ocr()
        # PaddleOCR v3: predict() returns list[OCRResult]; each result has
        # rec_texts, rec_scores, rec_polys keys.
        results = ocr.predict(bgr)

        blocks = []
        if results:
            res = results[0]
            for text, conf, polygon in zip(res["rec_texts"], res["rec_scores"], res["rec_polys"]):
                bbox = polygon_to_bbox(polygon)
                blocks.append({"text": text, "confidence": round(float(conf), 4), "bbox": bbox})

        return assemble_result(self.name, blocks, file_name)
