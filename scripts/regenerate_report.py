"""Rebuild ``evaluation_report.json`` from the saved per-item extractions.

This re-runs ONLY the metric/aggregation layer over the predictions already saved
under ``output/extractions/*.json`` — no OCR or LLM calls. Use it after changing
the evaluation/metric code so the committed report matches the current code shape
(without paying for a full re-run).

Usage:
    python scripts/regenerate_report.py
    python scripts/regenerate_report.py --extractions output/extractions --output output/evaluation_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from invoice_extractor.evaluation.metrics import aggregate_results, evaluate_single  # noqa: E402
from invoice_extractor.evaluation.report import print_summary, report_to_dict  # noqa: E402


def main(argv: list | None = None) -> None:
    parser = argparse.ArgumentParser(description="Rebuild report from saved extractions")
    parser.add_argument("--extractions", default="output/extractions", help="Dir of per-item JSON")
    parser.add_argument("--output", default="output/evaluation_report.json", help="Report path")
    args = parser.parse_args(argv)

    extractions_dir = Path(args.extractions)
    files = sorted(extractions_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"No extraction files found in {extractions_dir}")

    sample_results = []
    for fp in files:
        rec = json.loads(fp.read_text(encoding="utf-8"))
        pred = rec.get("extraction")
        gt_str = rec.get("ground_truth", "") or ""
        sample_id = rec.get("id", fp.stem)
        if pred is None or not gt_str:
            # Skip records that errored during the original run (no usable prediction).
            continue
        sample_results.append(evaluate_single(pred, gt_str, sample_id))

    report = aggregate_results(sample_results)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report_to_dict(report), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print_summary(report, out)


if __name__ == "__main__":
    main()
