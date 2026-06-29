from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, Any

import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    PROCESSED_FILE,
    MODELS_DIR,
    REPORTS_DIR,
    RANDOM_STATE,
    FEATURE_COLUMNS,
    TARGET_COL,
    TREATMENT_COL,
)
from src.features import make_X
from src.metrics import qini_auc, uplift_by_decile, targeting_policy_summary


def safe_qini(y_true, treatment, uplift_score) -> float | None:
    try:
        return float(qini_auc(y_true, treatment, uplift_score))
    except Exception:
        return None


def safe_auc(y_true, y_score) -> float | None:
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return None


def fit_propensity_model(X_train: pd.DataFrame, treatment_train: pd.Series):
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    max_iter=5000,
                    solver="lbfgs",
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model.fit(X_train, treatment_train)
    return model


def train_x_learner(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    treatment_col: str = TREATMENT_COL,
    test_size: float = 0.25,
) -> Dict[str, Any]:
    """
    Custom X-Learner implementation for binary marketing uplift.

    Steps:
    1. Train outcome models for treated and control groups.
    2. Impute treatment effects.
    3. Train treatment-effect models.
    4. Combine treatment effects using propensity scores.
    """

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=df[target_col],
    )

    treated_df = train_df[train_df[treatment_col] == 1].copy()
    control_df = train_df[train_df[treatment_col] == 0].copy()

    X_treated = make_X(treated_df)
    y_treated = treated_df[target_col].astype(int)

    X_control = make_X(control_df)
    y_control = control_df[target_col].astype(int)

    # Outcome models
    mu1_model = HistGradientBoostingClassifier(
        max_iter=180,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
    )
    mu0_model = HistGradientBoostingClassifier(
        max_iter=180,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
    )

    mu1_model.fit(X_treated, y_treated)
    mu0_model.fit(X_control, y_control)

    # Imputed treatment effects
    mu0_on_treated = mu0_model.predict_proba(X_treated)[:, 1]
    mu1_on_control = mu1_model.predict_proba(X_control)[:, 1]

    d_treated = y_treated.values - mu0_on_treated
    d_control = mu1_on_control - y_control.values

    # Effect models
    tau1_model = HistGradientBoostingRegressor(
        max_iter=180,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
    )
    tau0_model = HistGradientBoostingRegressor(
        max_iter=180,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
    )

    tau1_model.fit(X_treated, d_treated)
    tau0_model.fit(X_control, d_control)

    # Propensity model
    X_train = make_X(train_df)
    propensity_model = fit_propensity_model(X_train, train_df[treatment_col].astype(int))

    X_test = make_X(test_df)

    tau1 = tau1_model.predict(X_test)
    tau0 = tau0_model.predict(X_test)
    propensity = propensity_model.predict_proba(X_test)[:, 1]

    # X-Learner weighting
    uplift_x = propensity * tau0 + (1 - propensity) * tau1

    predictions = test_df.copy().reset_index(drop=True)
    predictions["uplift_x_learner"] = uplift_x
    predictions["x_learner_propensity"] = propensity
    predictions["recommend_treatment_x_learner"] = (predictions["uplift_x_learner"] > 0).astype(int)

    metrics = {
        "model": "custom_x_learner",
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "target_col": target_col,
        "treatment_col": treatment_col,
        "target_rate_test": float(predictions[target_col].mean()),
        "treatment_rate_test": float(predictions[treatment_col].mean()),
        "x_learner_qini_auc": safe_qini(
            predictions[target_col],
            predictions[treatment_col],
            predictions["uplift_x_learner"],
        ),
        "avg_predicted_uplift": float(predictions["uplift_x_learner"].mean()),
        "share_recommended_for_treatment": float(predictions["recommend_treatment_x_learner"].mean()),
        "top_decile_avg_uplift": float(
            predictions.sort_values("uplift_x_learner", ascending=False)
            .head(max(1, len(predictions) // 10))["uplift_x_learner"]
            .mean()
        ),
    }

    bundle = {
        "mu1_model": mu1_model,
        "mu0_model": mu0_model,
        "tau1_model": tau1_model,
        "tau0_model": tau0_model,
        "propensity_model": propensity_model,
        "feature_columns": FEATURE_COLUMNS,
        "target_col": target_col,
        "treatment_col": treatment_col,
        "metrics": metrics,
    }

    return {
        "bundle": bundle,
        "predictions": predictions,
        "metrics": metrics,
    }


def train_econml_causal_forest(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    treatment_col: str = TREATMENT_COL,
    test_size: float = 0.25,
) -> Dict[str, Any]:
    """
    EconML Causal Forest DML model.

    This estimates heterogeneous treatment effects:
    uplift = E[Y | treatment=1, X] - E[Y | treatment=0, X]
    """

    try:
        from econml.dml import CausalForestDML
    except ImportError as exc:
        raise ImportError(
            "EconML is not installed. Run: pip install econml"
        ) from exc

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=df[target_col],
    )

    X_train = make_X(train_df)
    X_test = make_X(test_df)

    Y_train = train_df[target_col].astype(float).values
    T_train = train_df[treatment_col].astype(int).values

    cf_model = CausalForestDML(
        model_y=RandomForestRegressor(
            n_estimators=100,
            min_samples_leaf=50,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        model_t=RandomForestClassifier(
            n_estimators=100,
            min_samples_leaf=50,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        discrete_treatment=True,
        n_estimators=300,
        min_samples_leaf=100,
        max_depth=None,
        random_state=RANDOM_STATE,
    )

    cf_model.fit(
        Y=Y_train,
        T=T_train,
        X=X_train,
    )

    uplift_cf = cf_model.effect(X_test)

    try:
        lower, upper = cf_model.effect_interval(X_test)
    except Exception:
        lower = np.full(len(X_test), np.nan)
        upper = np.full(len(X_test), np.nan)

    predictions = test_df.copy().reset_index(drop=True)
    predictions["uplift_causal_forest"] = uplift_cf
    predictions["uplift_causal_forest_lower"] = lower
    predictions["uplift_causal_forest_upper"] = upper
    predictions["recommend_treatment_causal_forest"] = (
        predictions["uplift_causal_forest"] > 0
    ).astype(int)

    metrics = {
        "model": "econml_causal_forest_dml",
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "target_col": target_col,
        "treatment_col": treatment_col,
        "target_rate_test": float(predictions[target_col].mean()),
        "treatment_rate_test": float(predictions[treatment_col].mean()),
        "causal_forest_qini_auc": safe_qini(
            predictions[target_col],
            predictions[treatment_col],
            predictions["uplift_causal_forest"],
        ),
        "avg_predicted_uplift": float(predictions["uplift_causal_forest"].mean()),
        "share_recommended_for_treatment": float(
            predictions["recommend_treatment_causal_forest"].mean()
        ),
        "top_decile_avg_uplift": float(
            predictions.sort_values("uplift_causal_forest", ascending=False)
            .head(max(1, len(predictions) // 10))["uplift_causal_forest"]
            .mean()
        ),
        "confident_positive_share": float(
            (predictions["uplift_causal_forest_lower"] > 0).mean()
        ),
        "confident_negative_share": float(
            (predictions["uplift_causal_forest_upper"] < 0).mean()
        ),
    }

    bundle = {
        "causal_forest_model": cf_model,
        "feature_columns": FEATURE_COLUMNS,
        "target_col": target_col,
        "treatment_col": treatment_col,
        "metrics": metrics,
    }

    return {
        "bundle": bundle,
        "predictions": predictions,
        "metrics": metrics,
    }


def train_causalml_x_learner_optional(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    treatment_col: str = TREATMENT_COL,
    test_size: float = 0.25,
) -> Dict[str, Any]:
    """
    Optional CausalML X-Learner version.

    Use this only if causalml installs successfully in your environment.
    """

    try:
        from causalml.inference.meta import BaseXClassifier
    except ImportError as exc:
        raise ImportError(
            "CausalML is not installed. Run: pip install causalml. "
            "If it fails, skip this optional version and use the custom X-Learner."
        ) from exc

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=df[target_col],
    )

    X_train = make_X(train_df).values
    y_train = train_df[target_col].astype(int).values
    treatment_train = train_df[treatment_col].astype(int).values

    X_test = make_X(test_df).values

    learner = BaseXClassifier(
        outcome_learner=RandomForestClassifier(
            n_estimators=100,
            min_samples_leaf=50,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        effect_learner=RandomForestRegressor(
            n_estimators=100,
            min_samples_leaf=50,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        control_name=0,
    )

    learner.fit(
        X=X_train,
        treatment=treatment_train,
        y=y_train,
    )

    uplift = learner.predict(X_test)

    if isinstance(uplift, pd.DataFrame):
        uplift_values = uplift.iloc[:, 0].values
    else:
        uplift_values = np.asarray(uplift).reshape(-1)

    predictions = test_df.copy().reset_index(drop=True)
    predictions["uplift_causalml_x_learner"] = uplift_values
    predictions["recommend_treatment_causalml_x_learner"] = (
        predictions["uplift_causalml_x_learner"] > 0
    ).astype(int)

    metrics = {
        "model": "causalml_x_learner",
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "target_col": target_col,
        "treatment_col": treatment_col,
        "causalml_x_learner_qini_auc": safe_qini(
            predictions[target_col],
            predictions[treatment_col],
            predictions["uplift_causalml_x_learner"],
        ),
        "avg_predicted_uplift": float(predictions["uplift_causalml_x_learner"].mean()),
        "share_recommended_for_treatment": float(
            predictions["recommend_treatment_causalml_x_learner"].mean()
        ),
    }

    bundle = {
        "causalml_x_learner": learner,
        "feature_columns": FEATURE_COLUMNS,
        "target_col": target_col,
        "treatment_col": treatment_col,
        "metrics": metrics,
    }

    return {
        "bundle": bundle,
        "predictions": predictions,
        "metrics": metrics,
    }


def save_advanced_artifacts(name: str, result: Dict[str, Any]) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / f"{name}_bundle.joblib"
    pred_path = REPORTS_DIR / f"{name}_predictions.csv"
    metrics_path = REPORTS_DIR / f"{name}_metrics.json"
    decile_path = REPORTS_DIR / f"{name}_decile_summary.csv"

    joblib.dump(result["bundle"], model_path)
    result["predictions"].to_csv(pred_path, index=False)

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(result["metrics"], f, indent=2)

    uplift_col = None
    for col in result["predictions"].columns:
        if col.startswith("uplift_"):
            uplift_col = col
            break

    if uplift_col:
        deciles = uplift_by_decile(
            result["predictions"][result["bundle"]["target_col"]],
            result["predictions"][result["bundle"]["treatment_col"]],
            result["predictions"][uplift_col],
        )
        deciles.to_csv(decile_path, index=False)

    print(f"Saved {name} model: {model_path}")
    print(f"Saved {name} predictions: {pred_path}")
    print(f"Saved {name} metrics: {metrics_path}")


def run_all_advanced_models(
    processed_file: Path = PROCESSED_FILE,
    target_col: str = TARGET_COL,
    treatment_col: str = TREATMENT_COL,
    max_rows: int | None = 250000,
    include_causalml: bool = False,
) -> None:
    df = pd.read_csv(processed_file)

    if max_rows is not None and len(df) > max_rows:
        df = df.sample(max_rows, random_state=RANDOM_STATE).reset_index(drop=True)

    print(f"Training advanced models on {len(df):,} rows")
    print(f"Target: {target_col}")
    print(f"Treatment: {treatment_col}")

    x_result = train_x_learner(df, target_col=target_col, treatment_col=treatment_col)
    save_advanced_artifacts("x_learner", x_result)

    cf_result = train_econml_causal_forest(df, target_col=target_col, treatment_col=treatment_col)
    save_advanced_artifacts("econml_causal_forest", cf_result)

    if include_causalml:
        causalml_result = train_causalml_x_learner_optional(
            df,
            target_col=target_col,
            treatment_col=treatment_col,
        )
        save_advanced_artifacts("causalml_x_learner", causalml_result)


if __name__ == "__main__":
    run_all_advanced_models()