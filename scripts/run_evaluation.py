"""Batch evaluation CLI (thin wrapper).

Usage:
    python scripts/run_evaluation.py [--split test] [--limit 20] [--type invoice]
                                     [--engine paddleocr] [--output evaluation_report.json]

All logic lives in ``invoice_extractor.evaluation.report``. When the package is
installed (``pip install -e .``) you can also run it via the ``invoice-eval``
console script.
"""

import sys
from pathlib import Path

# Allow running from a source checkout without installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from invoice_extractor.evaluation.report import main  # noqa: E402

if __name__ == "__main__":
    main()
