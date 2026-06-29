from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from src.config import FIGURES_DIR
from src.metrics import qini_curve


def create_qini_plot(predictions: pd.DataFrame, target_col: str, treatment_col: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    curve = qini_curve(predictions[target_col], predictions[treatment_col], predictions["uplift_t_learner"])
    curve.to_csv(FIGURES_DIR.parent / "qini_curve_points.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(curve["population_fraction"], curve["incremental_outcome"], label="T-Learner uplift ranking")
    plt.plot(curve["population_fraction"], curve["random_baseline"], linestyle="--", label="Random targeting baseline")
    plt.title("Qini-style Incremental Outcome Curve")
    plt.xlabel("Fraction of customers targeted")
    plt.ylabel("Estimated incremental outcomes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "qini_curve.png", dpi=160)
    plt.close()
