"""Unit tests for value normalization and comparison."""

import pytest

from invoice_extractor.evaluation.normalize import (
    compare_invoice_date,
    compare_invoice_no,
    compare_net_worth,
    compare_store_name,
    normalize_invoice_date,
    normalize_invoice_no,
    normalize_net_worth,
    parse_ground_truth,
    parse_receipt_gt,
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
