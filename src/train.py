from __future__ import annotations

import argparse
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

from src.config import (
    PROCESSED_FILE,
    MODEL_BUNDLE_FILE,
    METRICS_FILE,
    PREDICTIONS_FILE,
    REPORTS_DIR,
    TARGET_COL,
    TREATMENT_COL,
    OUTCOME_VALUE,
    CONTACT_COST,
    RANDOM_STATE,
    FEATURE_COLUMNS,
)
from src.features import make_X, make_s_learner_X, build_feature_pipeline
from src.metrics import qini_auc, uplift_by_decile, targeting_policy_summary
from src.evaluate import create_qini_plot
from src.utils import save_json


def train_campaignlift(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    treatment_col: str = TREATMENT_COL,
    test_size: float = 0.25,
) -> Dict[str, Any]:
    validate_columns(df, target_col, treatment_col)

    strata = df[target_col].astype(str) + "_" + df[treatment_col].astype(str)

    if strata.value_counts().min() < 2:
        strata = df[target_col]

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=strata,
    )

    # 1. Response model baseline: predicts outcome without causal separation.
    response_model = build_feature_pipeline(
        HistGradientBoostingClassifier(max_iter=120, learning_rate=0.06, random_state=RANDOM_STATE)
    )
    response_model.fit(make_X(train_df), train_df[target_col])

    # 2. S-Learner: one model with treatment as a feature.
    s_model = build_feature_pipeline(
        HistGradientBoostingClassifier(max_iter=160, learning_rate=0.06, random_state=RANDOM_STATE)
    )
    s_model.fit(make_s_learner_X(train_df, treatment_col), train_df[target_col])

    # 3. T-Learner: separate treated/control response models.
    treated_df = train_df[train_df[treatment_col] == 1]
    control_df = train_df[train_df[treatment_col] == 0]

    t_treated_model = build_feature_pipeline(
        HistGradientBoostingClassifier(max_iter=160, learning_rate=0.06, random_state=RANDOM_STATE)
    )
    t_control_model = build_feature_pipeline(
        HistGradientBoostingClassifier(max_iter=160, learning_rate=0.06, random_state=RANDOM_STATE)
    )
    t_treated_model.fit(make_X(treated_df), treated_df[target_col])
    t_control_model.fit(make_X(control_df), control_df[target_col])

    predictions, metrics = score_models(
        test_df=test_df,
        response_model=response_model,
        s_model=s_model,
        t_treated_model=t_treated_model,
        t_control_model=t_control_model,
        target_col=target_col,
        treatment_col=treatment_col,
    )

    deciles = uplift_by_decile(
        predictions[target_col],
        predictions[treatment_col],
        predictions["uplift_t_learner"],
    )
    deciles.to_csv(REPORTS_DIR / "decile_policy_summary.csv", index=False)

    policy = targeting_policy_summary(
        predictions[target_col],
        predictions[treatment_col],
        predictions["uplift_t_learner"],
        OUTCOME_VALUE,
        CONTACT_COST,
    )
    policy.to_csv(REPORTS_DIR / "targeting_policy_summary.csv", index=False)

    create_qini_plot(predictions, target_col, treatment_col)

    bundle = {
        "response_model": response_model,
        "s_model": s_model,
        "t_treated_model": t_treated_model,
        "t_control_model": t_control_model,
        "feature_columns": FEATURE_COLUMNS,
        "target_col": target_col,
        "treatment_col": treatment_col,
        "metrics": metrics,
    }
    return {"bundle": bundle, "metrics": metrics, "predictions": predictions}


def validate_columns(df: pd.DataFrame, target_col: str, treatment_col: str) -> None:
    required = set(FEATURE_COLUMNS + [target_col, treatment_col])
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df[target_col].nunique() < 2:
        raise ValueError(f"Target {target_col} has only one class. Increase sample_rows.")
    if df[treatment_col].nunique() < 2:
        raise ValueError(f"Treatment {treatment_col} has only one class. Increase sample_rows.")


def predict_s_learner_uplift(s_model, df: pd.DataFrame, treatment_col: str):
    treated = make_X(df)
    treated[treatment_col] = 1
    control = make_X(df)
    control[treatment_col] = 0
    p_treated = s_model.predict_proba(treated)[:, 1]
    p_control = s_model.predict_proba(control)[:, 1]
    return p_treated, p_control, p_treated - p_control


def predict_t_learner_uplift(t_treated_model, t_control_model, df: pd.DataFrame):
    X = make_X(df)
    p_treated = t_treated_model.predict_proba(X)[:, 1]
    p_control = t_control_model.predict_proba(X)[:, 1]
    return p_treated, p_control, p_treated - p_control


def score_models(test_df, response_model, s_model, t_treated_model, t_control_model, target_col, treatment_col):
    test = test_df.copy().reset_index(drop=True)
    X_test = make_X(test)

    response_prob = response_model.predict_proba(X_test)[:, 1]
    s_p_treated, s_p_control, s_uplift = predict_s_learner_uplift(s_model, test, treatment_col)
    t_p_treated, t_p_control, t_uplift = predict_t_learner_uplift(t_treated_model, t_control_model, test)

    test["response_probability"] = response_prob
    test["s_p_treated"] = s_p_treated
    test["s_p_control"] = s_p_control
    test["uplift_s_learner"] = s_uplift
    test["t_p_treated"] = t_p_treated
    test["t_p_control"] = t_p_control
    test["uplift_t_learner"] = t_uplift
    test["recommend_treatment"] = (test["uplift_t_learner"] > 0).astype(int)

    # Customer labels for storytelling.
    high_response = test[["t_p_treated", "t_p_control"]].max(axis=1) >= test[["t_p_treated", "t_p_control"]].max(axis=1).quantile(0.70)
    test["customer_type"] = np.select(
        [
            test["uplift_t_learner"] >= test["uplift_t_learner"].quantile(0.80),
            (test["uplift_t_learner"] < 0),
            high_response,
        ],
        ["Persuadable", "Sleeping Dog", "Sure Thing"],
        default="Lost Cause",
    )

    metrics = {
        "rows_test": int(len(test)),
        "target_col": target_col,
        "treatment_col": treatment_col,
        "target_rate_test": float(test[target_col].mean()),
        "treatment_rate_test": float(test[treatment_col].mean()),
        "response_model_roc_auc": safe_auc(test[target_col], response_prob),
        "response_model_average_precision": safe_average_precision(test[target_col], response_prob),
        "s_learner_qini_auc": qini_auc(test[target_col], test[treatment_col], s_uplift),
        "t_learner_qini_auc": qini_auc(test[target_col], test[treatment_col], t_uplift),
        "share_recommended_for_treatment": float(test["recommend_treatment"].mean()),
        "avg_predicted_t_learner_uplift": float(test["uplift_t_learner"].mean()),
        "top_decile_avg_predicted_uplift": float(test.sort_values("uplift_t_learner", ascending=False).head(max(1, len(test)//10))["uplift_t_learner"].mean()),
    }
    return test, metrics


def safe_auc(y_true, y_score):
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return None


def safe_average_precision(y_true, y_score):
    try:
        return float(average_precision_score(y_true, y_score))
    except ValueError:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default=TARGET_COL, choices=["visit", "conversion"])
    parser.add_argument("--treatment", default=TREATMENT_COL, choices=["treatment", "exposure"])
    args = parser.parse_args()

    df = pd.read_csv(PROCESSED_FILE)
    result = train_campaignlift(df, target_col=args.target, treatment_col=args.treatment)

    joblib.dump(result["bundle"], MODEL_BUNDLE_FILE)
    result["predictions"].to_csv(PREDICTIONS_FILE, index=False)
    save_json(result["metrics"], METRICS_FILE)

    print(f"Saved model bundle to {MODEL_BUNDLE_FILE}")
    print(f"Saved predictions to {PREDICTIONS_FILE}")
    print(result["metrics"])


if __name__ == "__main__":
    main()
