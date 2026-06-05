"""Unit tests for evaluation metrics."""

from invoice_extractor.evaluation.metrics import (
    SampleResult,
    aggregate_results,
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


def test_evaluate_single_receipt(receipt_pred, receipt_raw_data_str):
    # parsed_data has no header -> treated as receipt
    result = evaluate_single(receipt_pred, '{"json": "{\'store_name\': \'x\'}"}', "r-1", receipt_raw_data_str)
    assert result.document_type == "receipt"
    assert result.store_name_match is True
    assert result.total_match is True


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
