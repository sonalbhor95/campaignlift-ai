from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.config import FEATURE_COLUMNS


def make_X(df: pd.DataFrame) -> pd.DataFrame:
    return df[FEATURE_COLUMNS].copy()


def make_s_learner_X(df: pd.DataFrame, treatment_col: str) -> pd.DataFrame:
    X = make_X(df)
    X[treatment_col] = df[treatment_col].astype(int).values
    return X


def build_feature_pipeline(model):
    return Pipeline(steps=[("scaler", StandardScaler()), ("model", model)])
