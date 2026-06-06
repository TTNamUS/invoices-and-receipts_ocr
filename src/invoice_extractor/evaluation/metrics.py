"""Per-sample and aggregate evaluation metrics.

Critical invoice fields (invoice_no, invoice_date, total_net_worth) are scored
exactly as before. Optional fields and line items are scored additionally for the
"extract optional fields with high accuracy" bonus, for both invoices and receipts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from invoice_extractor.evaluation.normalize import (
    compare_amount,
    compare_exact,
    compare_invoice_date,
    compare_invoice_no,
    compare_net_worth,
    compare_store_name,
    compare_text,
    compare_total,
    parse_ground_truth,
    parse_receipt_ground_truth,
    parse_receipt_gt,
)

Comparator = Callable[[Optional[str], Optional[str]], bool]

# ── Optional-field specs: field name -> comparator ──────────────────────────
INVOICE_HEADER_OPTIONAL: dict[str, Comparator] = {
    "seller": compare_text,
    "client": compare_text,
    "seller_tax_id": compare_exact,
    "client_tax_id": compare_exact,
    "iban": compare_exact,
}
INVOICE_SUMMARY_OPTIONAL: dict[str, Comparator] = {
    "total_vat": compare_amount,
    "total_gross_worth": compare_amount,
}
RECEIPT_FIELDS: dict[str, Comparator] = {
    "store_name": compare_text,
    "store_addr": compare_text,
    "telephone": compare_exact,
    "date": compare_invoice_date,
    "time": compare_text,
    "subtotal": compare_amount,
    "tax": compare_amount,
    "total": compare_amount,
    "tips": compare_amount,
}


@dataclass
class SampleResult:
    sample_id: str
    document_type: str  # "invoice" | "receipt" | "error"

    invoice_no_match: Optional[bool] = None
    invoice_date_match: Optional[bool] = None
    net_worth_match: Optional[bool] = None
    all_critical_pass: Optional[bool] = None

    pred_invoice_no: Optional[str] = None
    gt_invoice_no: Optional[str] = None
    pred_invoice_date: Optional[str] = None
    gt_invoice_date: Optional[str] = None
    pred_net_worth: Optional[str] = None
    gt_net_worth: Optional[str] = None

    # Receipt fields (kept for backward-compat)
    pred_store_name: Optional[str] = None
    gt_store_name: Optional[str] = None
    store_name_match: Optional[bool] = None

    pred_total: Optional[str] = None
    gt_total: Optional[str] = None
    total_match: Optional[bool] = None

    pred_date: Optional[str] = None
    gt_date: Optional[str] = None

    # Optional-field accuracy (bonus). Only fields present in GT are recorded.
    # {field_name: bool} — True if the prediction matched.
    optional_field_matches: dict = field(default_factory=dict)
    # {"count_gt", "count_pred", "matched", "desc_match_rate", "value_match_rate"} or None
    line_item_metrics: Optional[dict] = None

    error: Optional[str] = None


@dataclass
class EvaluationReport:
    total_files_evaluated: int = 0
    total_invoices_evaluated: int = 0
    total_receipts_evaluated: int = 0
    total_errors: int = 0

    invoice_no_accuracy: float = 0.0
    invoice_date_accuracy: float = 0.0
    total_net_worth_accuracy: float = 0.0
    important_field_invoice_pass_rate: float = 0.0

    # Optional-field accuracy (bonus): {field_name: rate} over samples where GT has the field
    invoice_optional_accuracy: dict = field(default_factory=dict)
    receipt_field_accuracy: dict = field(default_factory=dict)

    # Line-item aggregate metrics: {"desc_match_rate", "value_match_rate", "count_accuracy"} or {}
    invoice_line_item_metrics: dict = field(default_factory=dict)
    receipt_line_item_metrics: dict = field(default_factory=dict)

    sample_results: list = field(default_factory=list)


def _str_or_none(value) -> Optional[str]:
    """Return str(value) only when value is not None."""
    return str(value) if value is not None else None


def _has_value(v) -> bool:
    """True when a GT value is present (not None / empty string)."""
    return v is not None and str(v).strip() != ""


def _score_optional(
    pred: dict, gt: dict, spec: dict[str, Comparator]
) -> dict:
    """Score each optional field whose GT value is present. Returns {field: bool}."""
    out: dict = {}
    for name, cmp in spec.items():
        gt_val = gt.get(name)
        if not _has_value(gt_val):
            continue
        out[name] = bool(cmp(_str_or_none(pred.get(name)), _str_or_none(gt_val)))
    return out


def evaluate_line_items(
    pred_items: list,
    gt_items: list,
    key_field: str,
    value_fields: list,
) -> Optional[dict]:
    """Greedy 1-1 matching of line items on ``key_field`` (description/name).

    Returns counts + per-matched-pair description and value match rates, or None
    when there is no GT to score against.

    - ``desc_match_rate``: matched pairs / len(gt_items)
    - ``value_match_rate``: avg over matched pairs of (value fields that match)
    """
    gt_items = gt_items or []
    pred_items = pred_items or []
    if not gt_items:
        return None

    remaining = list(range(len(pred_items)))
    matched_pairs: list[tuple[dict, dict]] = []
    for g in gt_items:
        g_key = _str_or_none(g.get(key_field))
        found = None
        for idx in remaining:
            if compare_text(_str_or_none(pred_items[idx].get(key_field)), g_key):
                found = idx
                break
        if found is not None:
            remaining.remove(found)
            matched_pairs.append((pred_items[found], g))

    matched = len(matched_pairs)
    desc_match_rate = matched / len(gt_items)

    # Value-field accuracy over matched pairs (amount-tolerant on numeric fields).
    value_scores: list[float] = []
    for p, g in matched_pairs:
        checks = []
        for vf in value_fields:
            if not _has_value(g.get(vf)):
                continue
            checks.append(compare_amount(_str_or_none(p.get(vf)), _str_or_none(g.get(vf))))
        if checks:
            value_scores.append(sum(checks) / len(checks))
    # None when no item had a scoreable value field in GT (distinct from 0.0 = all wrong).
    value_match_rate = round(sum(value_scores) / len(value_scores), 4) if value_scores else None

    return {
        "count_gt": len(gt_items),
        "count_pred": len(pred_items),
        "matched": matched,
        "desc_match_rate": round(desc_match_rate, 4),
        "value_match_rate": value_match_rate,
    }


def evaluate_single(
    pred_extraction: dict,
    ground_truth_str: str,
    sample_id: str,
    raw_data_str: str = "",
) -> SampleResult:
    """Compare a single LLM prediction against the dataset ground truth."""
    gt = parse_ground_truth(ground_truth_str)
    pred = pred_extraction if isinstance(pred_extraction, dict) else {}

    if gt is None:
        return _evaluate_receipt(pred, ground_truth_str, sample_id, raw_data_str)
    return _evaluate_invoice(pred, gt, sample_id)


def _evaluate_invoice(pred: dict, gt: dict, sample_id: str) -> SampleResult:
    pred_header = pred.get("header") or {}
    pred_summary = pred.get("summary") or {}
    gt_header = gt.get("header") or {}
    gt_summary = gt.get("summary") or {}

    pred_inv_no = pred_header.get("invoice_no")
    pred_inv_date = pred_header.get("invoice_date")
    pred_nw = pred_summary.get("total_net_worth")

    gt_inv_no = gt_header.get("invoice_no")
    gt_inv_date = gt_header.get("invoice_date")
    gt_nw = gt_summary.get("total_net_worth")

    no_match = compare_invoice_no(pred_inv_no, gt_inv_no)
    date_match = compare_invoice_date(pred_inv_date, gt_inv_date)
    nw_match = compare_net_worth(pred_nw, gt_nw)
    all_pass = no_match and date_match and nw_match

    # Optional fields (header + summary) + line items.
    optional = _score_optional(pred_header, gt_header, INVOICE_HEADER_OPTIONAL)
    optional.update(_score_optional(pred_summary, gt_summary, INVOICE_SUMMARY_OPTIONAL))
    line_items = evaluate_line_items(
        pred.get("items"),
        gt.get("items"),
        key_field="item_desc",
        value_fields=["item_qty", "item_net_price", "item_net_worth", "item_gross_worth"],
    )

    return SampleResult(
        sample_id=sample_id,
        document_type="invoice",
        invoice_no_match=no_match,
        invoice_date_match=date_match,
        net_worth_match=nw_match,
        all_critical_pass=all_pass,
        pred_invoice_no=_str_or_none(pred_inv_no),
        gt_invoice_no=_str_or_none(gt_inv_no),
        pred_invoice_date=_str_or_none(pred_inv_date),
        gt_invoice_date=_str_or_none(gt_inv_date),
        pred_net_worth=_str_or_none(pred_nw),
        gt_net_worth=_str_or_none(gt_nw),
        optional_field_matches=optional,
        line_item_metrics=line_items,
    )


def _evaluate_receipt(
    pred: dict, ground_truth_str: str, sample_id: str, raw_data_str: str
) -> SampleResult:
    # Prefer the canonical parsed_data.json receipt GT; fall back to raw_data labels.
    receipt_gt = parse_receipt_ground_truth(ground_truth_str)
    if receipt_gt is None:
        receipt_gt = parse_receipt_gt(raw_data_str) or {}

    optional = _score_optional(pred, receipt_gt, RECEIPT_FIELDS)
    line_items = evaluate_line_items(
        pred.get("line_items"),
        receipt_gt.get("line_items"),
        key_field="item_name",
        value_fields=["item_quantity", "item_value"],
    )

    pred_sn = _str_or_none(pred.get("store_name"))
    pred_tot = _str_or_none(pred.get("total"))
    pred_date = _str_or_none(pred.get("date"))
    gt_sn = _str_or_none(receipt_gt.get("store_name"))
    gt_tot = _str_or_none(receipt_gt.get("total"))
    gt_date = _str_or_none(receipt_gt.get("date"))

    return SampleResult(
        sample_id=sample_id,
        document_type=pred.get("document_type", "receipt"),
        pred_store_name=pred_sn,
        gt_store_name=gt_sn,
        store_name_match=compare_store_name(pred_sn, gt_sn),
        pred_total=pred_tot,
        gt_total=gt_tot,
        total_match=compare_total(pred_tot, gt_tot),
        pred_date=pred_date,
        gt_date=gt_date,
        optional_field_matches=optional,
        line_item_metrics=line_items,
    )


def _rate(matches: list) -> float:
    return sum(bool(m) for m in matches) / len(matches) if matches else 0.0


def _aggregate_optional(results: list) -> dict:
    """Per-field accuracy, averaged only over samples where the GT has the field."""
    buckets: dict = {}
    for r in results:
        for name, ok in r.optional_field_matches.items():
            buckets.setdefault(name, []).append(ok)
    return {name: round(_rate(vals), 4) for name, vals in buckets.items()}


def _aggregate_line_items(results: list) -> dict:
    """Average line-item rates over samples that have line-item metrics."""
    metrics = [r.line_item_metrics for r in results if r.line_item_metrics]
    if not metrics:
        return {}
    n = len(metrics)
    count_accuracy = sum(
        min(m["count_pred"], m["count_gt"]) / max(m["count_gt"], 1) for m in metrics
    ) / n
    value_rates = [m["value_match_rate"] for m in metrics if m["value_match_rate"] is not None]
    return {
        "samples_with_items": n,
        "desc_match_rate": round(sum(m["desc_match_rate"] for m in metrics) / n, 4),
        "value_match_rate": round(sum(value_rates) / len(value_rates), 4) if value_rates else None,
        "count_accuracy": round(count_accuracy, 4),
    }


def aggregate_results(sample_results: list) -> EvaluationReport:
    invoices = [r for r in sample_results if r.document_type == "invoice"]
    receipts = [r for r in sample_results if r.document_type == "receipt"]
    errors = [r for r in sample_results if r.document_type == "error"]

    n_inv = len(invoices)
    n_rec = len(receipts)

    report = EvaluationReport(
        total_files_evaluated=len(sample_results),
        total_invoices_evaluated=n_inv,
        total_receipts_evaluated=n_rec,
        total_errors=len(errors),
        sample_results=sample_results,
    )

    if n_inv > 0:
        report.invoice_no_accuracy = _rate([r.invoice_no_match for r in invoices])
        report.invoice_date_accuracy = _rate([r.invoice_date_match for r in invoices])
        report.total_net_worth_accuracy = _rate([r.net_worth_match for r in invoices])
        report.important_field_invoice_pass_rate = _rate([r.all_critical_pass for r in invoices])
        report.invoice_optional_accuracy = _aggregate_optional(invoices)
        report.invoice_line_item_metrics = _aggregate_line_items(invoices)

    if n_rec > 0:
        # store_name / total are reported via receipt_field_accuracy (which uses the
        # better comparators and excludes empty-GT samples). The standalone
        # *_match flags remain on each SampleResult for per-sample debugging.
        report.receipt_field_accuracy = _aggregate_optional(receipts)
        report.receipt_line_item_metrics = _aggregate_line_items(receipts)

    return report
