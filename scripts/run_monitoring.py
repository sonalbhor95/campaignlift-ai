from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.model_monitoring import create_monitoring_report
from src.config import PROCESSED_FILE, PREDICTIONS_FILE, TARGET_COL, TREATMENT_COL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-path", default=str(PROCESSED_FILE))
    parser.add_argument("--current-path", default=None)
    parser.add_argument("--predictions-path", default=str(PREDICTIONS_FILE))
    parser.add_argument("--target", default=TARGET_COL, choices=["visit", "conversion"])
    parser.add_argument("--treatment", default=TREATMENT_COL, choices=["treatment", "exposure"])

    args = parser.parse_args()

    create_monitoring_report(
        reference_path=Path(args.reference_path),
        current_path=Path(args.current_path) if args.current_path else None,
        predictions_path=Path(args.predictions_path) if args.predictions_path else None,
        target_col=args.target,
        treatment_col=args.treatment,
    )


if __name__ == "__main__":
    main()