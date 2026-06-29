from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import argparse

import joblib

from src.config import MODEL_BUNDLE_FILE, METRICS_FILE, PREDICTIONS_FILE, REPORTS_DIR
from src.data_ingestion import load_criteo_sample
from src.data_quality import run_quality_checks
from src.train import train_campaignlift
from src.utils import save_json


def main():
    parser = argparse.ArgumentParser(description="Run CampaignLift AI pipeline end-to-end.")
    parser.add_argument("--sample-rows", type=int, default=200000)
    parser.add_argument("--target", choices=["visit", "conversion"], default="visit")
    parser.add_argument("--treatment", choices=["treatment", "exposure"], default="treatment")
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    df = load_criteo_sample(args.sample_rows, force_download=args.force_download)
    print("Data ingestion complete.")

    quality = run_quality_checks(df)
    quality.to_csv(REPORTS_DIR / "data_quality_report.csv", index=False)
    print("Data quality report saved.")

    result = train_campaignlift(df, target_col=args.target, treatment_col=args.treatment)
    joblib.dump(result["bundle"], MODEL_BUNDLE_FILE)
    result["predictions"].to_csv(PREDICTIONS_FILE, index=False)
    save_json(result["metrics"], METRICS_FILE)
    print("Pipeline complete.")
    print(result["metrics"])


if __name__ == "__main__":
    main()
