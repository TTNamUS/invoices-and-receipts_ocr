"""Shared pytest fixtures: sample OCR text and ground-truth strings."""

import json

import pytest


@pytest.fixture
def invoice_pred() -> dict:
    """A correct invoice prediction matching INVOICE_GT below."""
    return {
        "document_type": "invoice",
        "header": {
            "invoice_no": "40378170",
            "invoice_date": "10/15/2012",
            "seller": "Patel, Thompson and Montgomery",
            "client": "Jackson, Odonnell and Jackson",
            "seller_tax_id": "958-74-3511",
            "client_tax_id": "998-87-7723",
            "iban": "GB77WRBQ31965128414006",
        },
        "items": [
            {
                "item_desc": "Leed's Wine Companion",
                "item_qty": "1,00",
                "item_net_price": "7,50",
                "item_net_worth": "7,50",
                "item_vat": "10%",
                "item_gross_worth": "8,25",
            }
        ],
        "summary": {
            "total_net_worth": "$7,50",
            "total_vat": "$ 0,75",
            "total_gross_worth": "$8,25",
        },
    }


@pytest.fixture
def invoice_gt_str() -> str:
    """Dataset ``parsed_data`` string: outer JSON wrapping inner Python-literal json."""
    inner = (
        "{'header': {'invoice_no': '40378170', 'invoice_date': '10/15/2012', "
        "'seller': 'Patel, Thompson and Montgomery', 'client': 'Jackson, Odonnell and Jackson', "
        "'seller_tax_id': '958-74-3511', 'client_tax_id': '998-87-7723', "
        "'iban': 'GB77WRBQ31965128414006'}, "
        "'items': [{'item_desc': \"Leed's Wine Companion\", 'item_qty': '1,00', "
        "'item_net_price': '7,50', 'item_net_worth': '7,50', 'item_vat': '10%', "
        "'item_gross_worth': '8,25'}], "
        "'summary': {'total_net_worth': '$7,50', 'total_vat': '$ 0,75', 'total_gross_worth': '$8,25'}}"
    )
    return json.dumps({"json": inner})


@pytest.fixture
def receipt_pred() -> dict:
    return {
        "document_type": "receipt",
        "store_name": "Walmart",
        "date": "01/02/2020",
        "total": "$12.34",
        "line_items": [{"item_name": "Milk", "item_quantity": "1", "item_value": "3.00"}],
    }


@pytest.fixture
def receipt_raw_data_str() -> str:
    """Dataset ``raw_data`` string with ocr_labels (used as receipt GT)."""
    labels = [
        {"label": "Store_name_value", "transcription": "Walmart"},
        {"label": "Total_value", "transcription": "12.34"},
        {"label": "Date_value", "transcription": "01/02/2020"},
    ]
    return json.dumps({"ocr_labels": labels})
