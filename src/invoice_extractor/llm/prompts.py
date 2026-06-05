"""System prompt, few-shot examples, and message builder for LLM extraction."""

SYSTEM_PROMPT = """You are a specialized document data extraction engine for invoices and receipts.

Extract structured data from OCR text and return ONLY valid JSON — no markdown fences, no explanation.

## For INVOICES (contain "Invoice no", "Date of issue", seller/client info, IBAN, line items with net/VAT/gross):

Return this exact schema:
{
  "document_type": "invoice",
  "header": {
    "invoice_no": "<string or null>",
    "invoice_date": "<string MM/DD/YYYY or null>",
    "seller": "<string or null>",
    "client": "<string or null>",
    "seller_tax_id": "<string or null>",
    "client_tax_id": "<string or null>",
    "iban": "<string or null>"
  },
  "items": [
    {
      "item_desc": "<string or null>",
      "item_qty": "<string or null>",
      "item_net_price": "<string or null>",
      "item_net_worth": "<string or null>",
      "item_vat": "<string or null>",
      "item_gross_worth": "<string or null>"
    }
  ],
  "summary": {
    "total_net_worth": "<string or null>",
    "total_vat": "<string or null>",
    "total_gross_worth": "<string or null>"
  }
}

## For RECEIPTS (store name, cashier, date/time, line items with prices, subtotal/tax/total):

Return this exact schema:
{
  "document_type": "receipt",
  "store_name": "<string or null>",
  "store_addr": "<string or null>",
  "telephone": "<string or null>",
  "date": "<string or null>",
  "time": "<string or null>",
  "subtotal": "<string or null>",
  "tax": "<string or null>",
  "total": "<string or null>",
  "tips": "<string or null>",
  "line_items": [
    {
      "item_name": "<string or null>",
      "item_quantity": "<string or null>",
      "item_value": "<string or null>"
    }
  ]
}

## Critical rules:
- Return ONLY the JSON object
- Use null (not empty string) for missing fields
- Preserve number formatting EXACTLY as found (e.g. "$ 44 364,64" stays "$ 44 364,64")
- invoice_date must be MM/DD/YYYY format
- invoice_no: digits only (strip any INV- prefix if present)
"""

# Few-shot example 1: single-item invoice
_FEW_SHOT_OCR_1 = """Invoice no: 40378170
Date of issue: 10/15/2012
Seller: Patel, Thompson and Montgomery 356 Kyle Vista New James, MA 46228
Client: Jackson, Odonnell and Jackson 267 John Track Suite 841 Jenniferville, PA 98601
Tax Id: 958-74-3511
Tax Id: 998-87-7723
IBAN: GB77WRBQ31965128414006
ITEMS No. Description Qty Net price Net worth VAT [%] Gross worth
1. Leed's Wine Companion Bottle Corkscrew Opener Gift Box Set with Foil Cutter 1,00 each 7,50 7,50 10% 8,25
SUMMARY
Net worth VAT [%] VAT Gross worth
7,50 10% 0,75 8,25
Total $7,50 $ 0,75 $8,25"""

_FEW_SHOT_ANSWER_1 = """{
  "document_type": "invoice",
  "header": {
    "invoice_no": "40378170",
    "invoice_date": "10/15/2012",
    "seller": "Patel, Thompson and Montgomery 356 Kyle Vista New James, MA 46228",
    "client": "Jackson, Odonnell and Jackson 267 John Track Suite 841 Jenniferville, PA 98601",
    "seller_tax_id": "958-74-3511",
    "client_tax_id": "998-87-7723",
    "iban": "GB77WRBQ31965128414006"
  },
  "items": [
    {
      "item_desc": "Leed's Wine Companion Bottle Corkscrew Opener Gift Box Set with Foil Cutter",
      "item_qty": "1,00",
      "item_net_price": "7,50",
      "item_net_worth": "7,50",
      "item_vat": "10%",
      "item_gross_worth": "8,25"
    }
  ],
  "summary": {
    "total_net_worth": "$7,50",
    "total_vat": "$ 0,75",
    "total_gross_worth": "$8,25"
  }
}"""

# Few-shot example 2: multi-item invoice
_FEW_SHOT_OCR_2 = """Invoice no: 46301857
Date of issue: 04/08/2017
Seller: Murphy-Sanders USNV Miller FPO AA 82012
Client: Parker, Graham and Swanson 607 Pearson Plaza New Jeffrey, NJ 42303
Tax Id: 917-83-2356
Tax Id: 900-94-4030
IBAN: GB24OZUP84517588674217
ITEMS Qty UM Net price VAT [%] No. Description Net worth Gross worth
1. Care & Repair of Furniture 5,00 each 4,25 21,25 10% 23,37
2. Harry Potter and the Chamber of Secrets (Book 2) by Rowling, J.K. Paperback 5,00 each 7,62 38,10 10% 41,91
SUMMARY
VAT [%] VAT Net worth Gross worth
10% 59,35 5,94 65,29
Total $ 59,35 $ 5,94 $ 65,29"""

_FEW_SHOT_ANSWER_2 = """{
  "document_type": "invoice",
  "header": {
    "invoice_no": "46301857",
    "invoice_date": "04/08/2017",
    "seller": "Murphy-Sanders USNV Miller FPO AA 82012",
    "client": "Parker, Graham and Swanson 607 Pearson Plaza New Jeffrey, NJ 42303",
    "seller_tax_id": "917-83-2356",
    "client_tax_id": "900-94-4030",
    "iban": "GB24OZUP84517588674217"
  },
  "items": [
    {
      "item_desc": "Care & Repair of Furniture",
      "item_qty": "5,00",
      "item_net_price": "4,25",
      "item_net_worth": "21,25",
      "item_vat": "10%",
      "item_gross_worth": "23,37"
    },
    {
      "item_desc": "Harry Potter and the Chamber of Secrets (Book 2) by Rowling, J.K. Paperback",
      "item_qty": "5,00",
      "item_net_price": "7,62",
      "item_net_worth": "38,10",
      "item_vat": "10%",
      "item_gross_worth": "41,91"
    }
  ],
  "summary": {
    "total_net_worth": "$ 59,35",
    "total_vat": "$ 5,94",
    "total_gross_worth": "$ 65,29"
  }
}"""


def build_messages(ocr_text: str) -> list:
    """Build multi-turn messages with 2 few-shot examples followed by the query.

    Few-shot examples are placed in conversation turns (not the system prompt)
    per LLM best practices.
    """
    return [
        {"role": "user", "content": f"Extract structured data from this OCR text:\n\n{_FEW_SHOT_OCR_1}"},
        {"role": "assistant", "content": _FEW_SHOT_ANSWER_1},
        {"role": "user", "content": f"Extract structured data from this OCR text:\n\n{_FEW_SHOT_OCR_2}"},
        {"role": "assistant", "content": _FEW_SHOT_ANSWER_2},
        {"role": "user", "content": f"Extract structured data from this OCR text:\n\n{ocr_text}"},
    ]
