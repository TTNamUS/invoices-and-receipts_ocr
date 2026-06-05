"""Streamlit demo: upload an invoice/receipt image, run OCR + LLM extraction."""

import io
import json
import sys
from pathlib import Path

# Allow running from a source checkout without installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from PIL import Image  # noqa: E402

from invoice_extractor.config import settings  # noqa: E402
from invoice_extractor.llm import LLMExtractor  # noqa: E402
from invoice_extractor.ocr import run_ocr  # noqa: E402

st.set_page_config(page_title="Invoice & Receipt Extraction", layout="wide")

st.title("Invoice & Receipt Extraction Demo")
st.caption("OCR + LLM pipeline — upload an invoice or receipt image to extract structured data")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")

    ocr_engine = st.selectbox(
        "OCR Engine",
        ["paddleocr", "tesseract"],
        index=0 if settings.ocr_engine == "paddleocr" else 1,
    )

    provider_options = ["openai", "gemini"]
    provider = st.selectbox(
        "LLM Provider",
        provider_options,
        index=provider_options.index(settings.llm_provider)
        if settings.llm_provider in provider_options
        else 0,
    )

    model_defaults = {
        "openai": [settings.openai_model, "gpt-5.4"],
        "gemini": [settings.gemini_model, "gemini/gemini-3.1-pro-preview"],
    }
    model_list = list(dict.fromkeys(model_defaults.get(provider, [settings.openai_model])))
    default_model = settings.llm_model if settings.llm_model in model_list else model_list[0]
    selected_model = st.selectbox("Model", model_list, index=model_list.index(default_model))

    st.divider()
    st.markdown("**About**")
    st.markdown("- PaddleOCR extracts text + bboxes")
    st.markdown("- LLM parses text → structured JSON")
    st.markdown("- Critical fields: invoice_no, invoice_date, total_net_worth")

# ── File Upload ───────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload invoice or receipt image",
    type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
)

if not uploaded_file:
    st.info("Upload an image to get started.")
    st.stop()

# ── OCR ───────────────────────────────────────────────────────────────────────
file_bytes = uploaded_file.getvalue()


@st.cache_data(show_spinner=False)
def cached_ocr(file_bytes: bytes, engine: str, fname: str) -> dict:
    img = Image.open(io.BytesIO(file_bytes))
    return run_ocr(img, engine_name=engine, file_name=fname)


with st.spinner(f"Running {ocr_engine.upper()} OCR..."):
    ocr_result = cached_ocr(file_bytes, ocr_engine, uploaded_file.name)

# ── Layout: image | OCR results ───────────────────────────────────────────────
col_img, col_ocr = st.columns([1, 2])

with col_img:
    st.subheader("Uploaded Document")
    st.image(file_bytes, width="stretch")

with col_ocr:
    st.subheader("Step 1 — OCR Results")

    m1, m2 = st.columns(2)
    m1.metric("Avg Confidence", f"{ocr_result['average_confidence']:.1%}")
    m2.metric("Text Blocks", len(ocr_result["text_blocks"]))

    with st.expander("Raw OCR Text", expanded=True):
        st.text_area(
            label="Raw OCR Text",
            value=ocr_result["raw_text"],
            height=220,
            label_visibility="collapsed",
        )

    with st.expander("Text Blocks (bbox + confidence)"):
        st.json(ocr_result["text_blocks"])

# ── LLM Extraction ────────────────────────────────────────────────────────────
st.divider()
st.subheader("Step 2 — LLM Structured Extraction")

if st.button("Run LLM Extraction", type="primary"):
    with st.spinner(f"Extracting with {selected_model} ..."):
        try:
            extractor = LLMExtractor(model=selected_model)
            result = extractor.extract(ocr_result["raw_text"])
            st.session_state["extraction_result"] = result
            st.session_state["extraction_error"] = None
        except Exception as exc:
            st.session_state["extraction_result"] = None
            st.session_state["extraction_error"] = str(exc)

if st.session_state.get("extraction_error"):
    st.error(f"Extraction failed: {st.session_state['extraction_error']}")

if st.session_state.get("extraction_result"):
    result = st.session_state["extraction_result"]
    doc_type = result.get("document_type", "unknown")

    st.info(f"Document type detected: **{doc_type.upper()}**")

    if doc_type == "invoice":
        header = result.get("header") or {}
        summary = result.get("summary") or {}
        items = result.get("items") or []

        col_h, col_s = st.columns(2)

        with col_h:
            st.markdown("**Header**")
            for k, v in header.items():
                label = k.replace("_", " ").title()
                st.text(f"{label}: {v}")

        with col_s:
            st.markdown("**Summary**")
            for k, v in summary.items():
                label = k.replace("_", " ").title()
                st.text(f"{label}: {v}")

        if items:
            st.markdown(f"**Line Items ({len(items)})**")
            df = pd.DataFrame(items)
            st.dataframe(df, width="stretch")

    elif doc_type == "receipt":
        receipt_fields = {k: v for k, v in result.items() if k not in ("document_type", "line_items")}
        line_items = result.get("line_items") or []

        col_r1, col_r2 = st.columns(2)
        items_list = list(receipt_fields.items())
        half = len(items_list) // 2

        with col_r1:
            for k, v in items_list[:half]:
                st.text(f"{k.replace('_', ' ').title()}: {v}")
        with col_r2:
            for k, v in items_list[half:]:
                st.text(f"{k.replace('_', ' ').title()}: {v}")

        if line_items:
            st.markdown(f"**Line Items ({len(line_items)})**")
            st.dataframe(pd.DataFrame(line_items), width="stretch")

    else:
        st.json(result)

    with st.expander("Full JSON Output"):
        st.json(result)

    st.download_button(
        label="Download JSON",
        data=json.dumps(result, indent=2, ensure_ascii=False),
        file_name=f"{uploaded_file.name}_extraction.json",
        mime="application/json",
    )
