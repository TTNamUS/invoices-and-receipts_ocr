"""OCR engine base class and shared geometry/image helpers."""

from __future__ import annotations

import io
import statistics
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

ImageInput = Union[Image.Image, str, Path, bytes]


def load_image(image_input: ImageInput) -> Image.Image:
    """Coerce any supported input into a PIL Image."""
    if isinstance(image_input, Image.Image):
        return image_input
    if isinstance(image_input, bytes):
        return Image.open(io.BytesIO(image_input))
    return Image.open(image_input)


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    """Convert a PIL RGB image to a numpy BGR array for PaddleOCR."""
    img = image.convert("RGB")
    arr = np.array(img)
    return arr[:, :, ::-1]  # RGB -> BGR


def polygon_to_bbox(polygon: list) -> list:
    """Convert a 4-corner polygon ``[[x,y],...]`` to ``[xmin, ymin, xmax, ymax]``."""
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


def sort_blocks_reading_order(blocks: list) -> list:
    """Sort text blocks in natural reading order (top-to-bottom, left-to-right)."""
    if not blocks:
        return blocks
    heights = [b["bbox"][3] - b["bbox"][1] for b in blocks]
    row_height = statistics.median(heights) if heights else 20
    row_height = max(row_height, 5)

    def sort_key(b):
        yc = (b["bbox"][1] + b["bbox"][3]) / 2
        xc = (b["bbox"][0] + b["bbox"][2]) / 2
        row = int(yc / row_height)
        return (row, xc)

    return sorted(blocks, key=sort_key)


def assemble_result(engine_name: str, blocks: list, file_name: str) -> dict:
    """Sort blocks, join raw text, and average confidence into the OCR dict."""
    blocks = sort_blocks_reading_order(blocks)
    raw_text = "\n".join(b["text"] for b in blocks)
    avg_conf = round(statistics.mean(b["confidence"] for b in blocks), 4) if blocks else 0.0
    return {
        "file_name": file_name,
        "ocr_engine": engine_name,
        "raw_text": raw_text,
        "average_confidence": avg_conf,
        "text_blocks": blocks,
    }


class OCREngine(ABC):
    """Abstract OCR engine. Implementations return the standard OCR dict."""

    name: str = "base"

    @abstractmethod
    def run(self, image_input: ImageInput, file_name: str = "unknown") -> dict:
        """Run OCR and return a dict with keys:
        ``file_name, ocr_engine, raw_text, average_confidence, text_blocks``.
        """
        raise NotImplementedError
