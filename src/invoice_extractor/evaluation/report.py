"""Batch evaluation runner, report serialization, and CLI entrypoint."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Optional

from invoice_extractor.config import settings
from invoice_extractor.evaluation.metrics import (
    EvaluationReport,
    SampleResult,
    aggregate_results,
    evaluate_single,
)
from invoice_extractor.evaluation.normalize import parse_ground_truth
from invoice_extractor.llm import LLMExtractor
from invoice_extractor.logging_config import get_logger
from invoice_extractor.ocr import run_ocr

logger = get_logger(__name__)


def _is_invoice(sample: dict) -> bool:
    return parse_ground_truth(sample.get("parsed_data", "")) is not None


def report_to_dict(report: EvaluationReport) -> dict:
    """Serialize an EvaluationReport to the JSON-ready dict shape."""
    return {
        "total_files_evaluated": report.total_files_evaluated,
        "total_invoices_evaluated": report.total_invoices_evaluated,
        "total_receipts_evaluated": report.total_receipts_evaluated,
        "total_errors": report.total_errors,
        "invoice_metrics": {
            "important_field_pass_rate": round(report.important_field_invoice_pass_rate, 4),
            "invoice_no_accuracy": round(report.invoice_no_accuracy, 4),
            "invoice_date_accuracy": round(report.invoice_date_accuracy, 4),
            "total_net_worth_accuracy": round(report.total_net_worth_accuracy, 4),
            "optional_field_accuracy": report.invoice_optional_accuracy,
            "line_item_metrics": report.invoice_line_item_metrics,
        },
        "receipt_metrics": {
            "field_accuracy": report.receipt_field_accuracy,
            "line_item_metrics": report.receipt_line_item_metrics,
        },
        "per_sample_details": [
            {
                "id": r.sample_id,
                "type": r.document_type,
                "invoice_no_match": r.invoice_no_match,
                "invoice_date_match": r.invoice_date_match,
                "net_worth_match": r.net_worth_match,
                "all_pass": r.all_critical_pass,
                "pred_invoice_no": r.pred_invoice_no,
                "gt_invoice_no": r.gt_invoice_no,
                "pred_invoice_date": r.pred_invoice_date,
                "gt_invoice_date": r.gt_invoice_date,
                "pred_net_worth": r.pred_net_worth,
                "gt_net_worth": r.gt_net_worth,
                "pred_store_name": r.pred_store_name,
                "gt_store_name": r.gt_store_name,
                "store_name_match": r.store_name_match,
                "pred_total": r.pred_total,
                "gt_total": r.gt_total,
                "total_match": r.total_match,
                "pred_date": r.pred_date,
                "gt_date": r.gt_date,
                "optional_field_matches": r.optional_field_matches,
                "line_item_metrics": r.line_item_metrics,
                "error": r.error,
            }
            for r in report.sample_results
        ],
    }


def dict_to_sample_result(d: dict) -> SampleResult:
    """Reconstruct a SampleResult from a ``per_sample_details`` entry.

    Used by ``--merge`` to keep previously-evaluated samples (of document types
    not being re-run) and re-aggregate them together with fresh results.
    """
    return SampleResult(
        sample_id=d.get("id"),
        document_type=d.get("type", "error"),
        invoice_no_match=d.get("invoice_no_match"),
        invoice_date_match=d.get("invoice_date_match"),
        net_worth_match=d.get("net_worth_match"),
        all_critical_pass=d.get("all_pass"),
        pred_invoice_no=d.get("pred_invoice_no"),
        gt_invoice_no=d.get("gt_invoice_no"),
        pred_invoice_date=d.get("pred_invoice_date"),
        gt_invoice_date=d.get("gt_invoice_date"),
        pred_net_worth=d.get("pred_net_worth"),
        gt_net_worth=d.get("gt_net_worth"),
        pred_store_name=d.get("pred_store_name"),
        gt_store_name=d.get("gt_store_name"),
        store_name_match=d.get("store_name_match"),
        pred_total=d.get("pred_total"),
        gt_total=d.get("gt_total"),
        total_match=d.get("total_match"),
        pred_date=d.get("pred_date"),
        gt_date=d.get("gt_date"),
        optional_field_matches=d.get("optional_field_matches") or {},
        line_item_metrics=d.get("line_item_metrics"),
        error=d.get("error"),
    )


def print_summary(report: EvaluationReport, output_path: Path) -> None:
    """Print the user-facing summary table to stdout."""
    print(f"\n{'=' * 50}")
    print(f"Evaluation complete -> {output_path}")
    print(f"  Total evaluated:          {report.total_files_evaluated}")
    print(f"  Invoices:                 {report.total_invoices_evaluated}")
    print(f"  Receipts:                 {report.total_receipts_evaluated}")
    print(f"  Errors:                   {report.total_errors}")
    if report.total_invoices_evaluated > 0:
        print("  [Invoice metrics]")
        print(f"    Pass rate (all 3 fields): {report.important_field_invoice_pass_rate:.1%}")
        print(f"    Invoice no accuracy:      {report.invoice_no_accuracy:.1%}")
        print(f"    Invoice date accuracy:    {report.invoice_date_accuracy:.1%}")
        print(f"    Net worth accuracy:       {report.total_net_worth_accuracy:.1%}")
        _print_optional("Invoice optional fields", report.invoice_optional_accuracy)
        _print_line_items("Invoice line items", report.invoice_line_item_metrics)
    if report.total_receipts_evaluated > 0:
        print("  [Receipt metrics]")
        _print_optional("Receipt fields", report.receipt_field_accuracy)
        _print_line_items("Receipt line items", report.receipt_line_item_metrics)


def _print_optional(title: str, field_accuracy: dict) -> None:
    if not field_accuracy:
        return
    print(f"    [{title}]")
    for name, rate in sorted(field_accuracy.items(), key=lambda x: -x[1]):
        print(f"      {name:18s}: {rate:.1%}")


def _print_line_items(title: str, metrics: dict) -> None:
    if not metrics:
        return
    vmr = metrics.get("value_match_rate")
    vmr_str = f"{vmr:.1%}" if vmr is not None else "n/a"
    print(f"    [{title}] over {metrics.get('samples_with_items', 0)} samples")
    print(f"      desc match rate:  {metrics.get('desc_match_rate', 0):.1%}")
    print(f"      value match rate: {vmr_str}")
    print(f"      count accuracy:   {metrics.get('count_accuracy', 0):.1%}")


def run_evaluation(
    split: str,
    limit,
    doc_type: str,
    engine: str,
    output_path: Path,
    merge: bool = False,
) -> EvaluationReport:
    """Run batch OCR+LLM evaluation over a dataset split and write the report.

    When ``merge`` is True and ``output_path`` already exists, samples in the
    existing report whose document type is NOT being re-run are carried over, so
    the report stays complete (e.g. re-run only receipts, keep prior invoices).
    """
    from datasets import load_from_disk
    from tqdm import tqdm

    logger.info("Loading dataset from %s ...", settings.dataset_path)
    ds = load_from_disk(str(settings.dataset_path))
    dataset = ds[split]

    # Filter by document type first, then apply limit.
    if doc_type in ("invoice", "receipt"):
        want_invoice = doc_type == "invoice"
        indices = [i for i, s in enumerate(dataset) if _is_invoice(s) == want_invoice]
        dataset = dataset.select(indices)
        logger.info("Filtered to %d %ss in split='%s'", len(dataset), doc_type, split)

    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))

    logger.info(
        "Evaluating %d samples | split=%s | type=%s | engine=%s | model=%s",
        len(dataset),
        split,
        doc_type,
        engine,
        settings.llm_model,
    )

    # Per-item LLM extraction results are saved next to the report, e.g.
    # output/extractions/<sample_id>.json — one file per document.
    extractions_dir = output_path.parent / "extractions"
    extractions_dir.mkdir(parents=True, exist_ok=True)

    extractor = LLMExtractor()
    sample_results = []

    for sample in tqdm(dataset, desc="Processing"):
        sample_id = sample["id"]
        image = sample["image"]
        parsed_data_str = sample["parsed_data"]

        ocr_result = None
        pred = None
        error = None
        try:
            ocr_result = run_ocr(image, engine_name=engine, file_name=str(sample_id))
            raw_text = ocr_result["raw_text"]
            pred = extractor.extract(raw_text)
            result = evaluate_single(pred, parsed_data_str, sample_id, sample.get("raw_data", ""))
        except Exception as exc:
            error = str(exc)
            logger.warning("Sample %s failed: %s", sample_id, exc)
            result = SampleResult(sample_id=sample_id, document_type="error", error=error)

        _save_extraction(
            extractions_dir,
            sample_id=sample_id,
            engine=engine,
            ocr_result=ocr_result,
            extraction=pred,
            ground_truth=parsed_data_str,
            error=error,
        )

        sample_results.append(result)
        time.sleep(0.05)  # small delay to avoid API rate limits

    if merge and output_path.exists():
        sample_results = _merge_with_existing(sample_results, doc_type, output_path)

    report = aggregate_results(sample_results)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report_to_dict(report), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print_summary(report, output_path)
    logger.info("Per-item extractions saved to %s/", extractions_dir)
    return report


def _merge_with_existing(fresh: list, doc_type: str, output_path: Path) -> list:
    """Combine freshly-run samples with carried-over samples from an existing report.

    Keeps existing samples whose type is NOT the one being re-run. Fresh results
    win on id collisions.
    """
    if doc_type not in ("invoice", "receipt"):
        # A full re-run replaces everything; nothing to carry over.
        return fresh
    try:
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        prior_details = existing.get("per_sample_details", [])
    except Exception as exc:
        logger.warning("Could not read existing report for --merge (%s); writing fresh only.", exc)
        return fresh

    fresh_ids = {str(r.sample_id) for r in fresh}
    carried = [
        dict_to_sample_result(d)
        for d in prior_details
        if d.get("type") != doc_type and str(d.get("id")) not in fresh_ids
    ]
    logger.info(
        "Merge: %d fresh %s sample(s) + %d carried-over sample(s) from %s",
        len(fresh), doc_type, len(carried), output_path.name,
    )
    return carried + fresh


def _save_extraction(
    extractions_dir: Path,
    *,
    sample_id,
    engine: str,
    ocr_result: Optional[dict],
    extraction: Optional[dict],
    ground_truth: str,
    error: Optional[str],
) -> None:
    """Persist a single sample's OCR text + LLM extraction + GT to its own JSON file."""
    record = {
        "id": sample_id,
        "ocr_engine": engine,
        "model": settings.llm_model,
        "raw_text": ocr_result["raw_text"] if ocr_result else None,
        "average_confidence": ocr_result["average_confidence"] if ocr_result else None,
        "extraction": extraction,
        "ground_truth": ground_truth,
        "error": error,
    }
    # Sanitize sample_id for use as a filename.
    safe_id = re.sub(r"[^\w.-]", "_", str(sample_id))
    path = extractions_dir / f"{safe_id}.json"
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")


def main(argv: list | None = None) -> None:
    parser = argparse.ArgumentParser(description="Batch OCR+LLM evaluation")
    parser.add_argument("--split", default=settings.eval_split, help="Dataset split: train/test/valid")
    parser.add_argument("--limit", type=int, default=None, help="Max samples to evaluate")
    parser.add_argument(
        "--type",
        dest="doc_type",
        default="all",
        choices=["all", "invoice", "receipt"],
        help="Filter by document type (default: all)",
    )
    parser.add_argument("--engine", default=settings.ocr_engine, help="OCR engine: paddleocr/tesseract")
    parser.add_argument("--output", default=str(settings.eval_report_path), help="Output JSON path")
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Update the existing --output report: re-run only --type, keep the other types.",
    )
    args = parser.parse_args(argv)

    run_evaluation(
        args.split, args.limit, args.doc_type, args.engine, Path(args.output), merge=args.merge
    )


if __name__ == "__main__":
    main()
