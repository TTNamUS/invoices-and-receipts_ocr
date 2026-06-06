"""Unit tests for value normalization and comparison."""

import pytest

from invoice_extractor.evaluation.normalize import (
    compare_amount,
    compare_exact,
    compare_invoice_date,
    compare_invoice_no,
    compare_net_worth,
    compare_store_name,
    compare_text,
    normalize_invoice_date,
    normalize_invoice_no,
    normalize_net_worth,
    parse_ground_truth,
    parse_receipt_gt,
    parse_receipt_ground_truth,
)


@pytest.mark.parametrize(
    "value,expected",
    [
        ("40378170", "40378170"),
        ("INV-40378170", "40378170"),
        ("Invoice No: 40378170", "40378170"),
        ("#40378170", "40378170"),
        ("No. 40378170", "40378170"),
        (None, None),
        ("", None),
        ("no digits", None),
    ],
)
def test_normalize_invoice_no(value, expected):
    assert normalize_invoice_no(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        ("10/15/2012", "10/15/2012"),
        ("2012-10-15", "10/15/2012"),
        ("October 15, 2012", "10/15/2012"),
        (None, None),
    ],
)
def test_normalize_invoice_date(value, expected):
    assert normalize_invoice_date(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        ("$ 44 364,64", 44364.64),
        ("$3172,99", 3172.99),
        ("$ 8,50", 8.50),
        ("7,50", 7.50),
        (None, None),
        ("n/a", None),
    ],
)
def test_normalize_net_worth(value, expected):
    assert normalize_net_worth(value) == expected


def test_compare_invoice_no_prefix_tolerant():
    assert compare_invoice_no("INV-40378170", "40378170") is True
    assert compare_invoice_no("40378171", "40378170") is False


def test_compare_invoice_date_format_tolerant():
    assert compare_invoice_date("2012-10-15", "10/15/2012") is True
    assert compare_invoice_date("10/16/2012", "10/15/2012") is False


def test_compare_net_worth_within_tolerance():
    assert compare_net_worth("$ 44 364,64", "44364.64") is True
    assert compare_net_worth("$ 44 364,65", "44364.64") is False  # outside 0.01


def test_compare_store_name_loose():
    assert compare_store_name("Wal-Mart", "walmart") is True
    assert compare_store_name("Target", "Walmart") is False
    assert compare_store_name(None, "Walmart") is False


def test_parse_ground_truth_invoice(invoice_gt_str):
    gt = parse_ground_truth(invoice_gt_str)
    assert gt is not None
    assert gt["header"]["invoice_no"] == "40378170"


def test_parse_ground_truth_receipt_returns_none():
    # No 'header' key -> treated as receipt -> None
    import json

    receipt_inner = "{'store_name': 'Walmart', 'total': '12.34'}"
    assert parse_ground_truth(json.dumps({"json": receipt_inner})) is None
    assert parse_ground_truth("") is None


def test_parse_receipt_gt(receipt_raw_data_str):
    gt = parse_receipt_gt(receipt_raw_data_str)
    assert gt == {"store_name": "Walmart", "total": "12.34", "date": "01/02/2020"}


def test_parse_receipt_ground_truth(receipt_gt_str):
    gt = parse_receipt_ground_truth(receipt_gt_str)
    assert gt is not None
    assert gt["store_name"] == "SPEEDWAY0006661"
    assert gt["total"] == "$26.09"
    assert len(gt["line_items"]) == 1


def test_parse_receipt_ground_truth_rejects_invoice(invoice_gt_str):
    # Invoice (has 'header') -> not a receipt GT
    assert parse_receipt_ground_truth(invoice_gt_str) is None
    assert parse_receipt_ground_truth("") is None


def test_compare_text_loose():
    assert compare_text("SPEEDWAY 0006661", "SPEEDWAY0006661") is True
    assert compare_text("LaPorte IN 46350", "LaPorteIN46350") is True
    assert compare_text("Acme Corp", "Globex") is False
    assert compare_text(None, "x") is False


def test_compare_text_rejects_short_fragment():
    # A small fragment must NOT match a long field (the old substring bug).
    assert compare_text("Acme", "Acme Corporation International Ltd") is False
    assert compare_text("Acme Corp", "Acme Corp 123 Main Street Suite 400 NY") is False


def test_compare_text_tolerates_ocr_noise():
    # Re-ordered tokens and a single dropped word still match (Jaccard >= 0.7).
    full = "Acme Corp 123 Main St"
    assert compare_text("Main St Acme Corp 123", full) is True
    assert compare_text("Acme Corp 123 Main", full) is True


def test_compare_store_name_rejects_fragment():
    assert compare_store_name("SPEED", "SPEEDWAY0006661") is False
    assert compare_store_name("SPEEDWAY 0006661", "SPEEDWAY0006661") is True


def test_compare_amount_rejects_one_cent():
    # 1-cent difference is a real mismatch now (tolerance tightened to 0.005).
    assert compare_amount("7,51", "7,50") is False
    assert compare_amount("7,50", "7,50") is True


def test_compare_net_worth_rejects_one_cent():
    assert compare_net_worth("7,51", "7,50") is False
    assert compare_net_worth("7,50", "7,50") is True


def test_compare_exact_normalized():
    assert compare_exact("958-74-3511", "958743511") is True
    assert compare_exact("GB77 WRBQ 3196", "GB77WRBQ3196") is True
    assert compare_exact("958-74-3512", "958-74-3511") is False


def test_compare_amount_tolerant():
    assert compare_amount("$26.09", "$ 26,09") is True
    assert compare_amount("$0.00", "0,00") is True
    assert compare_amount("$26.10", "$26.09") is False
    assert compare_amount(None, "1") is False


def test_compare_amount_us_format():
    # US format: comma = thousands, dot = decimal
    assert compare_amount("$30,473.00", "30473") is True
    assert compare_amount("$2,223.00", "2223.00") is True


def test_compare_amount_eu_space_thousands():
    # European space-grouped thousands separator
    assert compare_amount("$ 44 364,64", "44364.64") is True
    assert compare_amount("1 234 567,89", "1234567.89") is True


def test_compare_amount_repeated_tokens():
    # Receipt GT often repeats the amount (OCR captured it twice)
    assert compare_amount("89.60", "89.60 89.60") is True
    assert compare_amount("$1.65", "$0.83 $1.65 0.83$ 1.65$") is True
