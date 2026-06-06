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


def compare_net_worth(pred: Optional[str], gt: Optional[str], tolerance: float = 0.005) -> bool:
    """Compare two amounts. Both are rounded to 2 decimals by ``normalize_net_worth``,
    so a 0.005 tolerance means "equal to the cent" while absorbing float error — a
    1-cent difference (7.51 vs 7.50) correctly counts as a mismatch.
    """
    pv = normalize_net_worth(pred)
    gv = normalize_net_worth(gt)
    if pv is None or gv is None:
        return False
    return abs(pv - gv) < tolerance


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
    """Store-name match. Delegates to the length-gated ``compare_text`` so a tiny
    fragment no longer matches a long store name."""
    return compare_text(pred, gt)


def compare_total(pred: Optional[str], gt: Optional[str], tolerance: float = 0.005) -> bool:
    pv = normalize_net_worth(pred)
    gv = normalize_net_worth(gt)
    if pv is None or gv is None:
        return False
    return abs(pv - gv) < tolerance


def _parse_inner_json(parsed_data_str: str) -> Optional[dict]:
    """Parse the inner structured dict from a dataset ``parsed_data`` string.

    Outer layer is valid JSON; the inner 'json' field uses Python literal syntax.
    Returns the parsed dict (invoice or receipt) or None.
    """
    if not parsed_data_str:
        return None
    try:
        outer = json.loads(parsed_data_str)
        json_str = outer.get("json", "")
        if not json_str:
            return None
        inner = ast.literal_eval(json_str)
        return inner if isinstance(inner, dict) else None
    except Exception:
        return None


def parse_ground_truth(parsed_data_str: str) -> Optional[dict]:
    """Parse the invoice ground truth from the dataset's ``parsed_data`` field.

    Returns the invoice dict if it has a 'header' key, else None (receipt).
    """
    inner = _parse_inner_json(parsed_data_str)
    if inner is None or "header" not in inner:
        return None
    return inner


def parse_receipt_ground_truth(parsed_data_str: str) -> Optional[dict]:
    """Parse the receipt ground truth from the dataset's ``parsed_data`` field.

    Returns the receipt dict (no 'header' key) or None. This is the canonical,
    fully-structured receipt GT (store_name, store_addr, telephone, date, time,
    subtotal, tax, total, tips, line_items) — preferred over ``raw_data`` labels.
    """
    inner = _parse_inner_json(parsed_data_str)
    if inner is None or "header" in inner:
        return None
    return inner


# ── Generic comparators (for optional-field accuracy) ───────────────────────
def _normalize_loose(value: str) -> str:
    """Lowercase, strip all whitespace and non-word chars (loose text compare)."""
    return re.sub(r"[\s\W]", "", str(value)).lower()


def _tokenize(value: str) -> set:
    """Lowercase word tokens (alphanumeric), for token-overlap comparison."""
    return set(re.findall(r"[a-z0-9]+", str(value).lower()))


def compare_text(
    pred: Optional[str],
    gt: Optional[str],
    token_threshold: float = 0.7,
    substr_ratio: float = 0.6,
) -> bool:
    """Free-text match (seller, client, store_addr, item descriptions).

    Tolerant of OCR noise (re-ordered or a few missing words) but NOT of a
    prediction that only captures a small fragment. A match requires either:
      1. exact match after loose normalization, OR
      2. token-set overlap (Jaccard) >= ``token_threshold``, OR
      3. one string contains the other AND the shorter is >= ``substr_ratio`` of
         the longer (so "Acme" no longer matches a 60-char seller line).
    """
    if pred is None or gt is None:
        return False
    pn = _normalize_loose(pred)
    gn = _normalize_loose(gt)
    if not pn or not gn:
        return pn == gn
    if pn == gn:
        return True

    # Token-set overlap (Jaccard) — order-insensitive, robust to a few dropped words.
    pt, gt_ = _tokenize(pred), _tokenize(gt)
    if pt and gt_:
        jaccard = len(pt & gt_) / len(pt | gt_)
        if jaccard >= token_threshold:
            return True

    # Length-gated substring: the shorter must cover most of the longer.
    if pn in gn or gn in pn:
        short, long = (pn, gn) if len(pn) <= len(gn) else (gn, pn)
        if len(short) / len(long) >= substr_ratio:
            return True
    return False


def compare_exact(pred: Optional[str], gt: Optional[str]) -> bool:
    """Normalized exact match for IDs / IBAN / telephone (strip spaces+punct)."""
    if pred is None or gt is None:
        return False
    return _normalize_loose(pred) == _normalize_loose(gt)


def _parse_amount_token(tok: str) -> Optional[float]:
    """Parse a single amount token, detecting US vs European decimal style.

    Used only for optional/receipt amount comparison — handles both
    "$30,473.00" (US: comma thousands) and "44 364,64" (EU: comma decimal).
    The invoice critical path keeps using ``normalize_net_worth`` unchanged.
    """
    s = tok.replace("$", "").strip()
    s = re.sub(r"(\d)\s+(\d)", r"\1\2", s)  # collapse space thousands-sep
    has_dot = "." in s
    has_comma = "," in s
    if has_dot and has_comma:
        # The right-most separator is the decimal point.
        if s.rfind(".") > s.rfind(","):
            s = s.replace(",", "")  # US: 30,473.00
        else:
            s = s.replace(".", "").replace(",", ".")  # EU: 30.473,00
    elif has_comma:
        s = s.replace(",", ".")  # lone comma = decimal (EU)
    s = re.sub(r"[^\d.\-]", "", s)
    if s in ("", ".", "-"):
        return None
    try:
        return round(float(s), 2)
    except ValueError:
        return None


def _amount_candidates(value: Optional[str]) -> list:
    """Extract all distinct numeric amounts from a (possibly multi-token) string.

    Receipt GT often repeats the amount, e.g. "89.60 89.60" or
    "$0.83 $1.65 0.83$ 1.65$" (OCR captured it several times). We tokenize and
    parse each token, returning the unique rounded floats found.
    """
    if value is None:
        return []
    # Collapse a space-grouped thousands separator (digit, space, exactly 3 digits
    # followed by a non-digit/decimal), e.g. "44 364,64" -> "44364,64", BEFORE
    # tokenizing. This keeps a space-grouped amount as one token while genuinely
    # repeated amounts ("89.60 89.60") still split into separate tokens.
    s = str(value)
    while True:
        collapsed = re.sub(r"(\d)\s+(\d{3})(?=[.,]|\b)", r"\1\2", s)
        if collapsed == s:
            break
        s = collapsed
    tokens = s.split()
    if not tokens:
        return []
    seen: list = []
    for tok in tokens:
        v = _parse_amount_token(tok)
        if v is not None and v not in seen:
            seen.append(v)
    return seen


def compare_amount(pred: Optional[str], gt: Optional[str], tolerance: float = 0.005) -> bool:
    """Numeric amount match (subtotal, tax, totals, item prices).

    Amounts are rounded to 2 decimals, so a 0.005 tolerance means "equal to the
    cent". Handles multi-token GT amounts (repeated values): a match counts if any
    GT candidate amount equals any predicted candidate amount.
    """
    preds = _amount_candidates(pred)
    gts = _amount_candidates(gt)
    if not preds or not gts:
        return False
    return any(abs(p - g) < tolerance for p in preds for g in gts)
