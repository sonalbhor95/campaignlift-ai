from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import auc


def qini_curve(y_true, treatment, uplift_score) -> pd.DataFrame:
    """Create a Qini-style incremental gain curve.

    Rows are sorted by predicted uplift descending. At each prefix, incremental
    outcome is estimated as treated outcomes minus scaled control outcomes.
    """
    df = pd.DataFrame(
        {
            "y": np.asarray(y_true).astype(float),
            "treatment": np.asarray(treatment).astype(int),
            "uplift": np.asarray(uplift_score).astype(float),
        }
    ).sort_values("uplift", ascending=False).reset_index(drop=True)

    df["n"] = np.arange(1, len(df) + 1)
    df["cum_treated"] = df["treatment"].cumsum()
    df["cum_control"] = (1 - df["treatment"]).cumsum()
    df["cum_y_treated"] = (df["y"] * df["treatment"]).cumsum()
    df["cum_y_control"] = (df["y"] * (1 - df["treatment"])).cumsum()

    control_scaled = df["cum_y_control"] * df["cum_treated"] / df["cum_control"].replace(0, np.nan)
    df["incremental_outcome"] = (df["cum_y_treated"] - control_scaled).fillna(0)
    df["population_fraction"] = df["n"] / len(df)

    final_gain = df["incremental_outcome"].iloc[-1]
    df["random_baseline"] = df["population_fraction"] * final_gain
    return df


def qini_auc(y_true, treatment, uplift_score) -> float:
    curve = qini_curve(y_true, treatment, uplift_score)
    model_auc = auc(curve["population_fraction"], curve["incremental_outcome"])
    random_auc = auc(curve["population_fraction"], curve["random_baseline"])
    return float(model_auc - random_auc)


def uplift_by_decile(y_true, treatment, uplift_score) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "y": np.asarray(y_true).astype(float),
            "treatment": np.asarray(treatment).astype(int),
            "uplift": np.asarray(uplift_score).astype(float),
        }
    ).sort_values("uplift", ascending=False).reset_index(drop=True)
    df["decile"] = pd.qcut(df.index + 1, q=10, labels=False, duplicates="drop") + 1

    rows = []
    for decile, group in df.groupby("decile"):
        treated = group[group["treatment"] == 1]
        control = group[group["treatment"] == 0]
        treated_rate = treated["y"].mean() if len(treated) else 0.0
        control_rate = control["y"].mean() if len(control) else 0.0
        rows.append(
            {
                "decile": int(decile),
                "customers": int(len(group)),
                "avg_predicted_uplift": float(group["uplift"].mean()),
                "treated_response_rate": float(treated_rate),
                "control_response_rate": float(control_rate),
                "observed_uplift": float(treated_rate - control_rate),
            }
        )
    return pd.DataFrame(rows)


def targeting_policy_summary(y_true, treatment, uplift_score, outcome_value: float, contact_cost: float) -> pd.DataFrame:
    curve = qini_curve(y_true, treatment, uplift_score)
    rows = []
    for pct in [0.1, 0.2, 0.3, 0.4, 0.5, 1.0]:
        idx = max(int(len(curve) * pct) - 1, 0)
        targeted = idx + 1
        incremental = float(curve.loc[idx, "incremental_outcome"])
        revenue = incremental * outcome_value
        cost = targeted * contact_cost
        rows.append(
            {
                "target_fraction": pct,
                "targeted_customers": targeted,
                "estimated_incremental_outcomes": incremental,
                "estimated_incremental_revenue": revenue,
                "estimated_contact_cost": cost,
                "estimated_profit": revenue - cost,
            }
        )
    return pd.DataFrame(rows)
