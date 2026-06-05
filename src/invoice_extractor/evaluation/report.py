"""Batch evaluation runner, report serialization, and CLI entrypoint."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

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
        },
        "receipt_metrics": {
            "store_name_accuracy": round(report.receipt_store_name_accuracy, 4),
            "total_accuracy": round(report.receipt_total_accuracy, 4),
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
                "error": r.error,
            }
            for r in report.sample_results
        ],
    }


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
    if report.total_receipts_evaluated > 0:
        print("  [Receipt metrics]")
        print(f"    Store name accuracy:      {report.receipt_store_name_accuracy:.1%}")
        print(f"    Total accuracy:           {report.receipt_total_accuracy:.1%}")


def run_evaluation(
    split: str,
    limit,
    doc_type: str,
    engine: str,
    output_path: Path,
) -> EvaluationReport:
    """Run batch OCR+LLM evaluation over a dataset split and write the report."""
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

    extractor = LLMExtractor()
    sample_results = []

    for sample in tqdm(dataset, desc="Processing"):
        sample_id = sample["id"]
        image = sample["image"]
        parsed_data_str = sample["parsed_data"]

        try:
            ocr_result = run_ocr(image, engine_name=engine, file_name=str(sample_id))
            raw_text = ocr_result["raw_text"]
            pred = extractor.extract(raw_text)
            result = evaluate_single(pred, parsed_data_str, sample_id, sample.get("raw_data", ""))
        except Exception as exc:
            logger.warning("Sample %s failed: %s", sample_id, exc)
            result = SampleResult(sample_id=sample_id, document_type="error", error=str(exc))

        sample_results.append(result)
        time.sleep(0.05)  # small delay to avoid API rate limits

    report = aggregate_results(sample_results)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report_to_dict(report), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print_summary(report, output_path)
    return report


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
    args = parser.parse_args(argv)

    run_evaluation(args.split, args.limit, args.doc_type, args.engine, Path(args.output))


if __name__ == "__main__":
    main()
