"""Per-sample and aggregate evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from invoice_extractor.evaluation.normalize import (
    compare_invoice_date,
    compare_invoice_no,
    compare_net_worth,
    compare_store_name,
    compare_total,
    parse_ground_truth,
    parse_receipt_gt,
)


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

    # Receipt fields
    pred_store_name: Optional[str] = None
    gt_store_name: Optional[str] = None
    store_name_match: Optional[bool] = None

    pred_total: Optional[str] = None
    gt_total: Optional[str] = None
    total_match: Optional[bool] = None

    pred_date: Optional[str] = None
    gt_date: Optional[str] = None

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

    # Receipt accuracy (vs ground truth)
    receipt_store_name_accuracy: float = 0.0
    receipt_total_accuracy: float = 0.0

    sample_results: list = field(default_factory=list)


def _str_or_none(value) -> Optional[str]:
    """Return str(value) only when value is not None."""
    return str(value) if value is not None else None


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
        # Receipt — compare against GT from raw_data.ocr_labels
        receipt_gt = parse_receipt_gt(raw_data_str) or {}
        pred_sn = _str_or_none(pred.get("store_name"))
        pred_tot = _str_or_none(pred.get("total"))
        pred_date = _str_or_none(pred.get("date"))
        gt_sn = receipt_gt.get("store_name")
        gt_tot = receipt_gt.get("total")
        gt_date = receipt_gt.get("date")
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
        )

    # Invoice
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
    )


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
        report.invoice_no_accuracy = sum(r.invoice_no_match for r in invoices) / n_inv
        report.invoice_date_accuracy = sum(r.invoice_date_match for r in invoices) / n_inv
        report.total_net_worth_accuracy = sum(r.net_worth_match for r in invoices) / n_inv
        report.important_field_invoice_pass_rate = sum(r.all_critical_pass for r in invoices) / n_inv

    if n_rec > 0:
        report.receipt_store_name_accuracy = sum(bool(r.store_name_match) for r in receipts) / n_rec
        report.receipt_total_accuracy = sum(bool(r.total_match) for r in receipts) / n_rec

    return report
