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

from src.advanced_causal_models import run_all_advanced_models
from src.config import PROCESSED_FILE, TARGET_COL, TREATMENT_COL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-file", default=str(PROCESSED_FILE))
    parser.add_argument("--target", default=TARGET_COL, choices=["visit", "conversion"])
    parser.add_argument("--treatment", default=TREATMENT_COL, choices=["treatment", "exposure"])
    parser.add_argument("--max-rows", type=int, default=250000)
    parser.add_argument("--include-causalml", action="store_true")

    args = parser.parse_args()

    run_all_advanced_models(
        processed_file=Path(args.processed_file),
        target_col=args.target,
        treatment_col=args.treatment,
        max_rows=args.max_rows,
        include_causalml=args.include_causalml,
    )


if __name__ == "__main__":
    main()