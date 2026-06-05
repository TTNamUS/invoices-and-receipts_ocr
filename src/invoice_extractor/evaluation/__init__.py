"""Evaluation package — normalization, metrics, and reporting."""

from invoice_extractor.evaluation.metrics import (
    EvaluationReport,
    SampleResult,
    aggregate_results,
    evaluate_single,
)

__all__ = [
    "EvaluationReport",
    "SampleResult",
    "aggregate_results",
    "evaluate_single",
]
