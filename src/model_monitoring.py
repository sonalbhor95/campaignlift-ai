from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.config import (
    PROCESSED_FILE,
    PREDICTIONS_FILE,
    REPORTS_DIR,
    FEATURE_COLUMNS,
    TARGET_COL,
    TREATMENT_COL,
)


def population_stability_index(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    reference = pd.to_numeric(reference, errors="coerce").dropna()
    current = pd.to_numeric(current, errors="coerce").dropna()

    if len(reference) == 0 or len(current) == 0:
        return 0.0

    breakpoints = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))

    if len(breakpoints) < 3:
        return 0.0

    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    cur_counts, _ = np.histogram(current, bins=breakpoints)

    ref_pct = ref_counts / max(ref_counts.sum(), 1)
    cur_pct = cur_counts / max(cur_counts.sum(), 1)

    ref_pct = np.clip(ref_pct, 0.0001, None)
    cur_pct = np.clip(cur_pct, 0.0001, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def numeric_drift_check(reference: pd.Series, current: pd.Series) -> dict:
    ref = pd.to_numeric(reference, errors="coerce").dropna()
    cur = pd.to_numeric(current, errors="coerce").dropna()

    if len(ref) == 0 or len(cur) == 0:
        return {
            "ks_p_value": None,
            "psi": None,
            "drift_level": "unknown",
            "drift_detected": False,
        }

    _, p_value = ks_2samp(ref, cur)
    psi = population_stability_index(ref, cur)

    if psi >= 0.25:
        drift_level = "major"
    elif psi >= 0.10:
        drift_level = "moderate"
    else:
        drift_level = "low"

    return {
        "ks_p_value": float(p_value),
        "psi": float(psi),
        "drift_level": drift_level,
        "drift_detected": bool((p_value < 0.05) or (psi >= 0.10)),
    }


def summarize_prediction_monitoring(df: pd.DataFrame, target_col: str, treatment_col: str) -> dict:
    summary = {}

    if target_col in df.columns:
        summary["target_rate"] = float(df[target_col].mean())

    if treatment_col in df.columns:
        summary["treatment_rate"] = float(df[treatment_col].mean())

    uplift_cols = [col for col in df.columns if col.startswith("uplift_")]
    for col in uplift_cols:
        summary[f"{col}_mean"] = float(df[col].mean())
        summary[f"{col}_p10"] = float(df[col].quantile(0.10))
        summary[f"{col}_p50"] = float(df[col].quantile(0.50))
        summary[f"{col}_p90"] = float(df[col].quantile(0.90))
        summary[f"{col}_positive_share"] = float((df[col] > 0).mean())

    customer_type_cols = [col for col in df.columns if "customer_type" in col]
    for col in customer_type_cols:
        summary[f"{col}_distribution"] = df[col].value_counts(normalize=True).to_dict()

    return summary


def create_monitoring_report(
    reference_path: str | Path = PROCESSED_FILE,
    current_path: str | Path | None = None,
    predictions_path: str | Path | None = PREDICTIONS_FILE,
    target_col: str = TARGET_COL,
    treatment_col: str = TREATMENT_COL,
) -> None:
    reference = pd.read_csv(reference_path)

    if current_path is None:
        # For portfolio demo: simulate "current" data from a different sample.
        current = reference.sample(frac=0.25, random_state=99).reset_index(drop=True)
    else:
        current = pd.read_csv(current_path)

    drift_report = {}

    for feature in FEATURE_COLUMNS:
        if feature in reference.columns and feature in current.columns:
            drift_report[feature] = numeric_drift_check(reference[feature], current[feature])

    monitoring_summary = {
        "reference_rows": int(len(reference)),
        "current_rows": int(len(current)),
        "feature_drift": drift_report,
        "features_with_drift": [
            feature for feature, result in drift_report.items() if result["drift_detected"]
        ],
    }

    if predictions_path is not None and Path(predictions_path).exists():
        preds = pd.read_csv(predictions_path)
        monitoring_summary["prediction_monitoring"] = summarize_prediction_monitoring(
            preds,
            target_col=target_col,
            treatment_col=treatment_col,
        )

    output_path = REPORTS_DIR / "model_monitoring_report.json"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(monitoring_summary, f, indent=2)

    print(f"Model monitoring report saved to {output_path}")


if __name__ == "__main__":
    create_monitoring_report()