"""Pydantic models describing OCR output and LLM-extracted documents.

These mirror the dict shapes the pipeline already produces. They are used for
validation and serialization only — the runtime values are unchanged. Every
extracted field is ``Optional[str]`` (None for missing), matching the prompt's
"use null for missing fields" rule.
"""

from __future__ import annotations

from typing import Literal, Optional, Union

from pydantic import BaseModel, Field


# ── OCR ────────────────────────────────────────────────────────────────────
class TextBlock(BaseModel):
    text: str
    confidence: float
    bbox: list[int]  # [xmin, ymin, xmax, ymax]


class OCRResult(BaseModel):
    file_name: str
    ocr_engine: str
    raw_text: str
    average_confidence: float
    text_blocks: list[TextBlock] = Field(default_factory=list)


# ── Invoice ────────────────────────────────────────────────────────────────
class InvoiceHeader(BaseModel):
    invoice_no: Optional[str] = None
    invoice_date: Optional[str] = None
    seller: Optional[str] = None
    client: Optional[str] = None
    seller_tax_id: Optional[str] = None
    client_tax_id: Optional[str] = None
    iban: Optional[str] = None


class InvoiceItem(BaseModel):
    item_desc: Optional[str] = None
    item_qty: Optional[str] = None
    item_net_price: Optional[str] = None
    item_net_worth: Optional[str] = None
    item_vat: Optional[str] = None
    item_gross_worth: Optional[str] = None


class InvoiceSummary(BaseModel):
    total_net_worth: Optional[str] = None
    total_vat: Optional[str] = None
    total_gross_worth: Optional[str] = None


class Invoice(BaseModel):
    document_type: Literal["invoice"] = "invoice"
    header: InvoiceHeader = Field(default_factory=InvoiceHeader)
    items: list[InvoiceItem] = Field(default_factory=list)
    summary: InvoiceSummary = Field(default_factory=InvoiceSummary)


# ── Receipt ────────────────────────────────────────────────────────────────
class ReceiptLineItem(BaseModel):
    item_name: Optional[str] = None
    item_quantity: Optional[str] = None
    item_value: Optional[str] = None


class Receipt(BaseModel):
    document_type: Literal["receipt"] = "receipt"
    store_name: Optional[str] = None
    store_addr: Optional[str] = None
    telephone: Optional[str] = None
    date: Optional[str] = None
    time: Optional[str] = None
    subtotal: Optional[str] = None
    tax: Optional[str] = None
    total: Optional[str] = None
    tips: Optional[str] = None
    line_items: list[ReceiptLineItem] = Field(default_factory=list)


ExtractedDocument = Union[Invoice, Receipt]


def validate_extraction(data: dict) -> Optional[BaseModel]:
    """Best-effort validation of a parsed LLM dict into a typed model.

    Returns the validated model, or ``None`` if it doesn't match either schema.
    Non-fatal by design: callers fall back to the raw dict so runtime behavior
    (and therefore evaluation metrics) stays identical.
    """
    if not isinstance(data, dict):
        return None
    doc_type = data.get("document_type")
    try:
        if doc_type == "receipt":
            return Receipt.model_validate(data)
        # default to invoice when header is present or type says so
        if doc_type == "invoice" or "header" in data:
            return Invoice.model_validate(data)
    except Exception:
        return None
    return None
