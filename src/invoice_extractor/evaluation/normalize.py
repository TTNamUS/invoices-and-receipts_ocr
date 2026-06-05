"""Value normalization and comparison for evaluation, plus ground-truth parsing.

Comparison happens on normalized values (float amounts, dateutil dates, digit-only
invoice numbers) rather than raw strings, so cosmetic formatting differences don't
count as mismatches.
"""

import ast
import json
import re
from typing import Optional


def normalize_invoice_no(value: Optional[str]) -> Optional[str]:
    """Extract the digit-only invoice number.

    Handles prefixes like INV-, #, No., etc. Ground truth is always an
    8-digit string e.g. "40378170".
    """
    if not value:
        return None
    cleaned = re.sub(r"(?i)(inv[-\s#]?|invoice\s*no\.?\s*:?\s*|no\.?\s*:?\s*|#\s*)", "", str(value)).strip()
    digits = re.sub(r"\D", "", cleaned)
    return digits if digits else None


def normalize_invoice_date(value: Optional[str]) -> Optional[str]:
    """Parse to datetime and return MM/DD/YYYY.

    Ground truth is always MM/DD/YYYY e.g. "09/18/2015".
    """
    if not value:
        return None
    s = str(value).strip()
    if re.match(r"^\d{2}/\d{2}/\d{4}$", s):
        return s
    try:
        from dateutil import parser as dp

        dt = dp.parse(s, dayfirst=False)
        return dt.strftime("%m/%d/%Y")
    except Exception:
        return s


def normalize_net_worth(value: Optional[str]) -> Optional[float]:
    """Strip $, collapse space-as-thousands-separator, replace , -> ., parse float.

    Ground truth examples: "$ 44 364,64", "$3172,99", "$ 8,50".
    """
    if not value:
        return None
    s = str(value)
    s = s.replace("$", "").strip()
    # Collapse spaces between digits (thousands separator)
    s = re.sub(r"(\d)\s+(\d)", r"\1\2", s)
    # Replace European decimal comma -> period
    s = s.replace(",", ".")
    # Remove anything except digits, dot, minus
    s = re.sub(r"[^\d.\-]", "", s)
    try:
        return round(float(s), 2)
    except ValueError:
        return None


def compare_invoice_no(pred: Optional[str], gt: Optional[str]) -> bool:
    pn = normalize_invoice_no(pred)
    gn = normalize_invoice_no(gt)
    if pn is None or gn is None:
        return False
    return pn == gn


def compare_invoice_date(pred: Optional[str], gt: Optional[str]) -> bool:
    pd_ = normalize_invoice_date(pred)
    gd = normalize_invoice_date(gt)
    if pd_ is None or gd is None:
        return False
    return pd_ == gd


def compare_net_worth(pred: Optional[str], gt: Optional[str], tolerance: float = 0.01) -> bool:
    pv = normalize_net_worth(pred)
    gv = normalize_net_worth(gt)
    if pv is None or gv is None:
        return False
    return abs(pv - gv) <= tolerance


def parse_receipt_gt(raw_data_str: str) -> Optional[dict]:
    """Extract receipt ground truth from ``raw_data.ocr_labels``.

    Returns ``{'store_name', 'total', 'date'}`` or None if unparseable.
    """
    if not raw_data_str:
        return None
    try:
        raw = json.loads(raw_data_str)
        labels = raw.get("ocr_labels", [])
        if isinstance(labels, str):
            labels = ast.literal_eval(labels)
    except Exception:
        return None

    gt: dict = {"store_name": None, "total": None, "date": None}
    store_parts, total_parts, date_parts = [], [], []
    for item in labels:
        if not isinstance(item, dict):
            continue
        label = item.get("label", "")
        text = str(item.get("transcription", "")).strip()
        if not text:
            continue
        if label == "Store_name_value":
            store_parts.append(text)
        elif label == "Total_value":
            total_parts.append(text)
        elif label == "Date_value":
            date_parts.append(text)

    if store_parts:
        gt["store_name"] = " ".join(store_parts)
    if total_parts:
        gt["total"] = total_parts[-1]  # take last (grand total)
    if date_parts:
        gt["date"] = date_parts[-1]
    return gt


def _normalize_store(value: str) -> str:
    """Lowercase, strip spaces/punctuation for loose store name comparison."""
    return re.sub(r"[\s\W]", "", value).lower()


def compare_store_name(pred: Optional[str], gt: Optional[str]) -> bool:
    """Match if normalized pred equals normalized gt, or either contains the other."""
    if pred is None or gt is None:
        return False
    pn = _normalize_store(pred)
    gn = _normalize_store(gt)
    return pn == gn or pn in gn or gn in pn


def compare_total(pred: Optional[str], gt: Optional[str], tolerance: float = 0.05) -> bool:
    pv = normalize_net_worth(pred)
    gv = normalize_net_worth(gt)
    if pv is None or gv is None:
        return False
    return abs(pv - gv) <= tolerance


def parse_ground_truth(parsed_data_str: str) -> Optional[dict]:
    """Parse the dataset's ``parsed_data`` field.

    Outer layer is valid JSON; the inner 'json' field uses Python literal syntax.
    Returns the invoice dict if it has a 'header' key, else None (receipt).
    """
    if not parsed_data_str:
        return None
    try:
        outer = json.loads(parsed_data_str)
        json_str = outer.get("json", "")
        if not json_str:
            return None
        inner = ast.literal_eval(json_str)
        if not isinstance(inner, dict) or "header" not in inner:
            return None
        return inner
    except Exception:
        return None
