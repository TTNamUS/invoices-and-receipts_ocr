"""Unit tests for Pydantic schema validation."""

from invoice_extractor.schemas import Invoice, Receipt, validate_extraction


def test_invoice_round_trip(invoice_pred):
    model = Invoice.model_validate(invoice_pred)
    assert model.document_type == "invoice"
    assert model.header.invoice_no == "40378170"
    assert len(model.items) == 1
    # Round-trip preserves the critical values
    dumped = model.model_dump()
    assert dumped["summary"]["total_net_worth"] == "$7,50"


def test_receipt_round_trip(receipt_pred):
    model = Receipt.model_validate(receipt_pred)
    assert model.document_type == "receipt"
    assert model.store_name == "Walmart"
    assert len(model.line_items) == 1


def test_validate_extraction_invoice(invoice_pred):
    model = validate_extraction(invoice_pred)
    assert isinstance(model, Invoice)


def test_validate_extraction_receipt(receipt_pred):
    model = validate_extraction(receipt_pred)
    assert isinstance(model, Receipt)


def test_validate_extraction_infers_invoice_from_header():
    # No document_type, but header present -> Invoice
    model = validate_extraction({"header": {"invoice_no": "1"}})
    assert isinstance(model, Invoice)


def test_validate_extraction_returns_none_for_garbage():
    assert validate_extraction({"foo": "bar"}) is None
    assert validate_extraction("not a dict") is None


def test_missing_fields_default_to_none():
    model = Invoice.model_validate({"document_type": "invoice"})
    assert model.header.invoice_no is None
    assert model.summary.total_net_worth is None
    assert model.items == []
