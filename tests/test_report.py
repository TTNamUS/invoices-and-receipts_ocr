"""Unit tests for the report layer: dict round-trip and --merge logic."""

from invoice_extractor.evaluation.metrics import aggregate_results, evaluate_single
from invoice_extractor.evaluation.report import (
    _merge_with_existing,
    dict_to_sample_result,
    report_to_dict,
)


def test_report_dict_roundtrip(invoice_pred, invoice_gt_str):
    r = evaluate_single(invoice_pred, invoice_gt_str, "inv-1")
    report = aggregate_results([r])
    d = report_to_dict(report)
    # Reconstruct the sample and confirm key fields survive the round-trip.
    back = dict_to_sample_result(d["per_sample_details"][0])
    assert back.sample_id == "inv-1"
    assert back.document_type == "invoice"
    assert back.all_critical_pass is True
    assert back.optional_field_matches["seller"] is True


def test_report_dict_has_no_legacy_receipt_keys(receipt_pred_full, receipt_gt_str):
    r = evaluate_single(receipt_pred_full, receipt_gt_str, "r-1")
    d = report_to_dict(aggregate_results([r]))
    rm = d["receipt_metrics"]
    assert "field_accuracy" in rm
    assert "line_item_metrics" in rm
    # Legacy aggregate keys are gone (subsumed by field_accuracy).
    assert "store_name_accuracy" not in rm
    assert "total_accuracy" not in rm


def test_merge_keeps_other_type(tmp_path, invoice_pred, invoice_gt_str, receipt_pred_full, receipt_gt_str):
    # 1) Write an initial report with one invoice + one receipt.
    inv = evaluate_single(invoice_pred, invoice_gt_str, "inv-1")
    rec = evaluate_single(receipt_pred_full, receipt_gt_str, "r-1")
    out = tmp_path / "report.json"
    import json

    out.write_text(json.dumps(report_to_dict(aggregate_results([inv, rec]))), encoding="utf-8")

    # 2) Simulate re-running ONLY receipts (a fresh receipt result for r-1).
    fresh_rec = evaluate_single(receipt_pred_full, receipt_gt_str, "r-1")
    merged = _merge_with_existing([fresh_rec], doc_type="receipt", output_path=out)

    # The old invoice is carried over; the receipt is the fresh one.
    types = sorted(r.document_type for r in merged)
    assert types == ["invoice", "receipt"]
    ids = {r.sample_id for r in merged}
    assert ids == {"inv-1", "r-1"}


def test_merge_all_is_noop(tmp_path, invoice_pred, invoice_gt_str):
    inv = evaluate_single(invoice_pred, invoice_gt_str, "inv-1")
    out = tmp_path / "report.json"
    import json

    out.write_text(json.dumps(report_to_dict(aggregate_results([inv]))), encoding="utf-8")
    fresh = [evaluate_single(invoice_pred, invoice_gt_str, "inv-1")]
    # doc_type="all" -> full re-run, nothing carried over.
    merged = _merge_with_existing(fresh, doc_type="all", output_path=out)
    assert merged == fresh
