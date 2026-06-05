"""Tesseract OCR engine (fallback). Requires the Tesseract binary installed."""

from __future__ import annotations

import statistics

from invoice_extractor.ocr.base import (
    ImageInput,
    OCREngine,
    assemble_result,
    load_image,
)


class TesseractOCREngine(OCREngine):
    name = "tesseract"

    def run(self, image_input: ImageInput, file_name: str = "unknown") -> dict:
        import pytesseract
        from pytesseract import Output

        image = load_image(image_input)
        data = pytesseract.image_to_data(image, output_type=Output.DICT)

        # Group words into lines.
        lines: dict = {}
        n = len(data["text"])
        for i in range(n):
            word = data["text"][i].strip()
            conf = int(data["conf"][i])
            if not word or conf < 0:
                continue
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            if key not in lines:
                lines[key] = {"words": [], "confs": [], "x": [], "y": [], "w": [], "h": []}
            lines[key]["words"].append(word)
            lines[key]["confs"].append(conf / 100.0)
            lines[key]["x"].append(data["left"][i])
            lines[key]["y"].append(data["top"][i])
            lines[key]["w"].append(data["width"][i])
            lines[key]["h"].append(data["height"][i])

        blocks = []
        for key in sorted(lines.keys()):
            ln = lines[key]
            text = " ".join(ln["words"])
            conf = round(statistics.mean(ln["confs"]), 4)
            x0 = min(ln["x"])
            y0 = min(ln["y"])
            x1 = max(xi + wi for xi, wi in zip(ln["x"], ln["w"]))
            y1 = max(yi + hi for yi, hi in zip(ln["y"], ln["h"]))
            blocks.append({"text": text, "confidence": conf, "bbox": [x0, y0, x1, y1]})

        return assemble_result(self.name, blocks, file_name)
