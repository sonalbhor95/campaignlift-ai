from __future__ import annotations

from typing import Dict, Any

import joblib
import pandas as pd

from src.config import MODEL_BUNDLE_FILE, FEATURE_COLUMNS


def load_model_bundle(path=MODEL_BUNDLE_FILE) -> Dict[str, Any]:
    return joblib.load(path)


def classify_customer(p_treated: float, p_control: float, uplift: float) -> str:
    if uplift > 0.01:
        return "Persuadable"
    if uplift < -0.005:
        return "Sleeping Dog"
    if max(p_treated, p_control) > 0.20:
        return "Sure Thing"
    return "Lost Cause"


def predict_customer(customer: Dict[str, Any], bundle: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if bundle is None:
        bundle = load_model_bundle()

    X = pd.DataFrame([customer])[FEATURE_COLUMNS]

    response_probability = float(bundle["response_model"].predict_proba(X)[0, 1])
    p_treated = float(bundle["t_treated_model"].predict_proba(X)[0, 1])
    p_control = float(bundle["t_control_model"].predict_proba(X)[0, 1])
    uplift = p_treated - p_control

    return {
        "response_probability": response_probability,
        "p_outcome_if_treated": p_treated,
        "p_outcome_if_control": p_control,
        "predicted_uplift": uplift,
        "recommend_treatment": uplift > 0,
        "customer_type": classify_customer(p_treated, p_control, uplift),
    }


if __name__ == "__main__":
    sample = {f"f{i}": 0.0 for i in range(12)}
    print(predict_customer(sample))
