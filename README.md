# Intelligent Invoice & Receipt Extraction

End-to-end document AI pipeline: **OCR → LLM → Structured JSON** with evaluation and a web demo.

```
Image → OCR (PaddleOCR / Tesseract) → raw text + bboxes → LLM (OpenAI / Gemini) → structured JSON
                                                                                       ↓
                                                       Evaluation vs ground-truth parsed_data
```

## Project Structure

```
src/invoice_extractor/        # installable package
├── config.py                 # Pydantic settings (env-driven)
├── logging_config.py         # logging setup
├── schemas.py                # Pydantic models (Invoice, Receipt, OCRResult, …)
├── pipeline.py               # orchestrator: image → OCR → LLM
├── ocr/                      # OCR engines
│   ├── base.py               #   OCREngine ABC + shared geometry/image helpers
│   ├── paddle_engine.py      #   PaddleOCR (primary)
│   └── tesseract_engine.py   #   Tesseract (fallback)
├── llm/                      # LLM extraction
│   ├── extractor.py          #   LiteLLM wrapper + retry/backoff + JSON parsing
│   └── prompts.py            #   system prompt
└── evaluation/               # evaluation
    ├── normalize.py          #   value normalization + comparison + GT parsing
    ├── metrics.py            #   per-sample + aggregate metrics
    └── report.py             #   batch runner + report serialization + CLI

web/app.py                    # Streamlit demo
scripts/run_evaluation.py     # CLI wrapper → evaluation.report:main
tests/                        # pytest unit tests (no network/OCR needed)
pyproject.toml                # packaging + deps + pytest/ruff config
```

## Setup

### 1. Install

```bash
# PaddleOCR (Windows CPU) — install paddlepaddle first
pip install paddlepaddle -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html

# Install the package (editable) with dev extras
pip install -e ".[dev]"
```

> Tesseract (optional fallback): install the binary from https://github.com/UB-Mannheim/tesseract/wiki

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env with your API key
```

OpenAI:
```
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-5.4-mini
```

Gemini:
```
LLM_PROVIDER=gemini
GOOGLE_API_KEY=AIza...
GEMINI_MODEL=gemini/gemini-3.5-flash
```

### 3. Dataset

The dataset lives at `dataset/invoices-and-receipts_ocr_v1/` (gitignored). No additional setup needed.

## Usage

### Web Demo

```bash
streamlit run web/app.py
```

Upload an invoice or receipt image → view OCR text + confidence → click **Run LLM Extraction** for structured JSON.

### Batch Evaluation

```bash
# Quick smoke test (5 samples)
python scripts/run_evaluation.py --split test --limit 5

# Full test set
python scripts/run_evaluation.py --split test

# Tesseract instead of PaddleOCR
python scripts/run_evaluation.py --split test --engine tesseract

# Custom output path
python scripts/run_evaluation.py --split test --output output/eval.json

# Re-run ONLY receipts and update the existing report (keeps the prior invoice results)
python scripts/run_evaluation.py --split test --type receipt --merge
```

The report is written to `output/evaluation_report.json` by default, and **each document's LLM
extraction** (OCR text + extracted JSON + ground truth) is saved to
`output/extractions/<id>.json` for inspection.

After `pip install -e .` you can also use the console script: `invoice-eval --split test --limit 5`.

### Run on Google Colab (CPU / GPU)

To run the evaluation or the web demo on Colab, open
[`notebooks/colab_eval.ipynb`](notebooks/colab_eval.ipynb). It clones this repo, auto-detects
GPU, installs the right `paddlepaddle` build, and runs the same pipeline. The notebook's own
markdown cells document each step.

### Run Tests

```bash
pytest
```

Tests cover normalization/comparison, evaluation metrics, schema validation, and message builder — no live API or OCR calls required.

### Use as a library

```python
from invoice_extractor import run_pipeline

result = run_pipeline("path/to/invoice.jpg", engine="paddleocr")
print(result["ocr"]["average_confidence"])
print(result["extraction"])   # structured dict
```

## Submission Answers

**1. Explain the OCR × LLM pipeline.**
An image goes to an OCR engine (PaddleOCR primary, Tesseract fallback) which returns text
blocks with per-block confidence and bounding boxes. Blocks are sorted into natural reading
order (top-to-bottom, left-to-right) and joined into raw text. That text is sent to an LLM
(OpenAI or Gemini via LiteLLM) with a schema-constrained system prompt; the LLM returns structured JSON, which is parsed and validated against Pydantic schemas. See `src/invoice_extractor/pipeline.py`.

**2. Accuracy on the important fields.**
Run `python scripts/run_evaluation.py --split test`. The report shape:
```json
{
  "total_files_evaluated": 125,
  "total_invoices_evaluated": 26,
  "total_receipts_evaluated": 99,
  "total_errors": 0,
  "invoice_metrics": {
    "important_field_pass_rate": 1.0,
    "invoice_no_accuracy": 1.0,
    "invoice_date_accuracy": 1.0,
    "total_net_worth_accuracy": 1.0,
    "optional_field_accuracy": {
      "seller": 0.7308,
      "client": 0.6923,
      "seller_tax_id": 1.0,
      "client_tax_id": 1.0,
      "iban": 0.6538,
      "total_vat": 1.0,
      "total_gross_worth": 0.9615
    },
    "line_item_metrics": {
      "samples_with_items": 26,
      "desc_match_rate": 0.9859,
      "value_match_rate": 1.0,
      "count_accuracy": 1.0
    }
  },
  "receipt_metrics": {
    "field_accuracy": {
      "store_name": 0.6465,
      "store_addr": 0.5125,
      "date": 0.6625,
      "time": 0.679,
      "subtotal": 0.8765,
      "tax": 0.8267,
      "total": 0.8791,
      "telephone": 0.75,
      "tips": 0.6667
    },
    "line_item_metrics": {
      "samples_with_items": 95,
      "desc_match_rate": 0.6836,
      "value_match_rate": 0.5408,
      "count_accuracy": 0.8819
    }
  }
}
```
An invoice **passes** only if all three critical fields match: `invoice_no`, `invoice_date`,
`total_net_worth`. The numbers above are the actual `output/evaluation_report.json` for the full
`test` split (run with the default config) — all 26 invoices pass on the three critical fields.

**3. Improvement techniques used.**
- **Prompt engineering** — explicit JSON schema in the system prompt with field-level rules.
- **JSON schema-guided extraction** — separate invoice vs receipt schemas + Pydantic validation.
- **Reading-order sort** — OCR blocks sorted by (row, x) before joining as raw text.
- **Value normalization** — comparison on normalized values (float amounts, dateutil dates,
  digit-only invoice numbers), not raw strings.

**4. Bonus.**
- **Optional-field accuracy** — per-field accuracy for *all* optional fields, on both document
  types, scored against the canonical `parsed_data.json` ground truth:
  - *Invoice*: `seller`, `client`, `seller_tax_id`, `client_tax_id`, `iban`, `total_vat`,
    `total_gross_worth`.
  - *Receipt*: `store_name`, `store_addr`, `telephone`, `date`, `time`, `subtotal`, `tax`,
    `total`, `tips`.
- **Line-item metric** — greedy 1-1 matching of line items on the description/name field, then
  per-pair value accuracy (qty / prices). Reported as `desc_match_rate`, `value_match_rate`,
  `count_accuracy`.
- **Robust amount normalization** — handles US (`$30,473.00`), European space-grouped
  (`$ 44 364,64`), and repeated-token OCR amounts (`89.60 89.60`) — only on the optional-field
  path, leaving the critical-field metrics untouched.
- Also: layout-aware reading order from bounding boxes; dual invoice/receipt extraction; typed
  Pydantic schemas; retry/backoff; logging; a unit-test suite; and a Colab runner.

The optional-field and line-item blocks appear in `evaluation_report.json` under
`invoice_metrics.optional_field_accuracy` / `line_item_metrics` and the analogous
`receipt_metrics.*`, and are printed in the CLI summary.

## Evaluation Notes

- **Critical-field metrics** are computed over **invoices only** (~26 of 125 test samples) and are
  unchanged: an invoice passes only if `invoice_no`, `invoice_date`, `total_net_worth` all match.
- **Optional-field accuracy** is averaged only over samples whose ground truth actually has the
  field (a blank GT field is neither a free pass nor a penalty).
- **Receipt GT** uses the canonical `parsed_data.json` (falls back to `raw_data.ocr_labels` if
  absent) — 9 scalar fields + line items, vs. the previous 3-field check.
- Normalization: invoice_no → digits only; dates → MM/DD/YYYY; amounts → float (±0.01); free-text fields → loose match (case/space/punct-insensitive, substring-tolerant).

## Example OCR Output

```json
{
  "file_name": "invoice_001.jpg",
  "ocr_engine": "paddleocr",
  "raw_text": "Invoice no: 40378170\nDate of issue: 10/15/2012\n...",
  "average_confidence": 0.9823,
  "text_blocks": [
    {"text": "Invoice no: 40378170", "confidence": 0.9912, "bbox": [42, 120, 380, 148]}
  ]
}
```

## Example LLM JSON Output

```json
{
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
      "item_desc": "Leed's Wine Companion Bottle Corkscrew Opener Gift Box Set",
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
}
```
