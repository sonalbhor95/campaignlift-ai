from __future__ import annotations

import pandas as pd

from src.config import PROCESSED_FILE, REPORTS_DIR, FEATURE_COLUMNS


def run_quality_checks(df: pd.DataFrame) -> pd.DataFrame:
    checks = []

    def add_check(name: str, value, status: str, details: str = ""):
        checks.append({"check": name, "value": value, "status": status, "details": details})

    add_check("row_count", len(df), "PASS" if len(df) > 10000 else "WARN", "Use at least 200k rows for a portfolio run.")
    add_check("missing_values", int(df.isna().sum().sum()), "PASS" if df.isna().sum().sum() == 0 else "WARN")

    for col in ["treatment", "exposure", "visit", "conversion"]:
        invalid = int((~df[col].isin([0, 1])).sum())
        add_check(f"invalid_binary_{col}", invalid, "PASS" if invalid == 0 else "FAIL")
        add_check(f"mean_{col}", float(df[col].mean()), "PASS")

    for col in FEATURE_COLUMNS:
        add_check(f"feature_std_{col}", float(df[col].std()), "PASS" if df[col].std() > 0 else "WARN")

    return pd.DataFrame(checks)


def main():
    df = pd.read_csv(PROCESSED_FILE)
    report = run_quality_checks(df)
    out = REPORTS_DIR / "data_quality_report.csv"
    report.to_csv(out, index=False)
    print(report)
    print(f"Saved data quality report to {out}")


if __name__ == "__main__":
    main()
