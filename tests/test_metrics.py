"""Unit tests for evaluation metrics."""

from invoice_extractor.evaluation.metrics import (
    SampleResult,
    aggregate_results,
    evaluate_line_items,
    evaluate_single,
)


def test_evaluate_single_invoice_all_pass(invoice_pred, invoice_gt_str):
    result = evaluate_single(invoice_pred, invoice_gt_str, "sample-1")
    assert result.document_type == "invoice"
    assert result.invoice_no_match is True
    assert result.invoice_date_match is True
    assert result.net_worth_match is True
    assert result.all_critical_pass is True


def test_evaluate_single_invoice_one_field_wrong(invoice_pred, invoice_gt_str):
    bad = {**invoice_pred, "summary": {**invoice_pred["summary"], "total_net_worth": "$999,00"}}
    result = evaluate_single(bad, invoice_gt_str, "sample-2")
    assert result.net_worth_match is False
    assert result.all_critical_pass is False  # any wrong field fails the invoice


def test_evaluate_single_receipt_from_raw_data(receipt_pred, receipt_raw_data_str):
    # No usable parsed_data json -> falls back to raw_data.ocr_labels GT
    result = evaluate_single(receipt_pred, "", "r-1", receipt_raw_data_str)
    assert result.document_type == "receipt"
    assert result.store_name_match is True
    assert result.total_match is True


def test_evaluate_single_receipt_from_parsed_data(receipt_pred_full, receipt_gt_str):
    # parsed_data.json present -> use it as the canonical receipt GT
    result = evaluate_single(receipt_pred_full, receipt_gt_str, "r-2")
    assert result.document_type == "receipt"
    ofm = result.optional_field_matches
    # Present GT fields should be scored; absent ones (telephone, tips) skipped.
    assert ofm["store_name"] is True
    assert ofm["store_addr"] is True
    assert ofm["date"] is True
    assert ofm["total"] is True
    assert ofm["subtotal"] is True
    assert ofm["tax"] is True
    assert "telephone" not in ofm  # GT value is empty -> not scored
    assert "tips" not in ofm


def test_invoice_optional_fields_scored(invoice_pred, invoice_gt_str):
    result = evaluate_single(invoice_pred, invoice_gt_str, "inv-opt")
    ofm = result.optional_field_matches
    assert ofm["seller"] is True
    assert ofm["client"] is True
    assert ofm["seller_tax_id"] is True
    assert ofm["iban"] is True
    assert ofm["total_vat"] is True
    assert ofm["total_gross_worth"] is True


def test_line_items_exact():
    gt = [{"item_desc": "Widget", "item_qty": "2", "item_net_worth": "10,00"}]
    pred = [{"item_desc": "Widget", "item_qty": "2", "item_net_worth": "10.00"}]
    m = evaluate_line_items(pred, gt, "item_desc", ["item_qty", "item_net_worth"])
    assert m["count_gt"] == 1 and m["matched"] == 1
    assert m["desc_match_rate"] == 1.0
    assert m["value_match_rate"] == 1.0


def test_line_items_reordered_and_missing():
    gt = [
        {"item_desc": "Alpha", "item_value": "1,00"},
        {"item_desc": "Beta", "item_value": "2,00"},
    ]
    pred = [{"item_desc": "Beta", "item_value": "2,00"}]  # one missing, order differs
    m = evaluate_line_items(pred, gt, "item_desc", ["item_value"])
    assert m["count_gt"] == 2 and m["count_pred"] == 1
    assert m["matched"] == 1
    assert m["desc_match_rate"] == 0.5


def test_line_items_none_when_no_gt():
    assert evaluate_line_items([{"item_desc": "x"}], [], "item_desc", []) is None


def test_aggregate_optional_excludes_missing_gt(receipt_pred_full, receipt_gt_str):
    r = evaluate_single(receipt_pred_full, receipt_gt_str, "r")
    report = aggregate_results([r])
    fa = report.receipt_field_accuracy
    assert fa["store_name"] == 1.0
    assert "telephone" not in fa  # never scored -> not in denominator


def test_aggregate_results_rates(invoice_pred, invoice_gt_str):
    pass_result = evaluate_single(invoice_pred, invoice_gt_str, "ok")
    bad = {**invoice_pred, "header": {**invoice_pred["header"], "invoice_no": "00000000"}}
    fail_result = evaluate_single(bad, invoice_gt_str, "bad")

    report = aggregate_results([pass_result, fail_result])
    assert report.total_invoices_evaluated == 2
    assert report.important_field_invoice_pass_rate == 0.5
    assert report.invoice_no_accuracy == 0.5
    assert report.invoice_date_accuracy == 1.0
    assert report.total_net_worth_accuracy == 1.0


def test_aggregate_counts_errors():
    err = SampleResult(sample_id="e", document_type="error", error="boom")
    report = aggregate_results([err])
    assert report.total_errors == 1
    assert report.total_invoices_evaluated == 0
