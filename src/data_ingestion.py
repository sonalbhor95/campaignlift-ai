from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse


import pandas as pd
import requests

from src.config import (
    CRITEO_URL,
    CRITEO_RAW_FILE,
    PROCESSED_FILE,
    REQUIRED_COLUMNS,
    DEFAULT_SAMPLE_ROWS,
    RANDOM_STATE,
)


def download_criteo(force: bool = False) -> Path:
    """
    Download the public Criteo uplift gzip file.

    The file is large. If it already exists locally, it is reused.
    """
    CRITEO_RAW_FILE.parent.mkdir(parents=True, exist_ok=True)

    if CRITEO_RAW_FILE.exists() and not force:
        print(f"Using existing file: {CRITEO_RAW_FILE}")
        return CRITEO_RAW_FILE

    print(f"Downloading Criteo uplift dataset from {CRITEO_URL}")

    with requests.get(CRITEO_URL, stream=True, timeout=60) as response:
        response.raise_for_status()
        with open(CRITEO_RAW_FILE, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    print(f"Saved raw gzip file to {CRITEO_RAW_FILE}")
    return CRITEO_RAW_FILE


def _clean_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize columns, keep required fields, convert values to numeric,
    and drop invalid rows.
    """
    chunk.columns = [str(c).strip() for c in chunk.columns]

    missing = sorted(set(REQUIRED_COLUMNS) - set(chunk.columns))
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    chunk = chunk[REQUIRED_COLUMNS].copy()

    for col in REQUIRED_COLUMNS:
        chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

    chunk = chunk.dropna().reset_index(drop=True)

    for col in ["treatment", "conversion", "visit", "exposure"]:
        chunk[col] = chunk[col].astype(int)

    return chunk


def load_criteo_sample(
    sample_rows: int = DEFAULT_SAMPLE_ROWS,
    force_download: bool = False,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """
    Load a treatment/control-balanced sample from the Criteo uplift dataset.

    Why this is needed:
    The raw Criteo file can be ordered in a way where the first N rows contain
    only one treatment class. Uplift modeling requires both treatment=0 and
    treatment=1, so this function streams the file in chunks and collects rows
    from both groups.
    """

    raw_path = download_criteo(force=force_download)

    target_control_rows = sample_rows // 2
    target_treated_rows = sample_rows - target_control_rows

    samples = {
        0: [],
        1: [],
    }

    collected = {
        0: 0,
        1: 0,
    }

    print("Creating stratified treatment/control sample...")

    for chunk_number, chunk in enumerate(
        pd.read_csv(raw_path, compression="gzip", chunksize=chunksize),
        start=1,
    ):
        chunk = _clean_chunk(chunk)

        for treatment_value, target_rows in [
            (0, target_control_rows),
            (1, target_treated_rows),
        ]:
            remaining = target_rows - collected[treatment_value]

            if remaining <= 0:
                continue

            group = chunk[chunk["treatment"] == treatment_value]

            if group.empty:
                continue

            take_n = min(remaining, len(group))

            sampled_group = group.sample(
                n=take_n,
                random_state=RANDOM_STATE + chunk_number + treatment_value,
            )

            samples[treatment_value].append(sampled_group)
            collected[treatment_value] += take_n

        print(
            f"Chunk {chunk_number}: "
            f"control={collected[0]:,}/{target_control_rows:,}, "
            f"treated={collected[1]:,}/{target_treated_rows:,}"
        )

        if collected[0] >= target_control_rows and collected[1] >= target_treated_rows:
            break

    if collected[0] == 0 or collected[1] == 0:
        raise ValueError(
            "Could not collect both treatment classes. "
            f"Collected control={collected[0]}, treated={collected[1]}. "
            "Try increasing sample_rows or verify the raw dataset."
        )

    df = pd.concat(samples[0] + samples[1], ignore_index=True)

    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_FILE, index=False)

    print(f"Saved processed sample to {PROCESSED_FILE}")
    print("Treatment distribution:")
    print(df["treatment"].value_counts())
    print("Outcome means:")
    print(df[["treatment", "exposure", "visit", "conversion"]].mean())

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-rows", type=int, default=DEFAULT_SAMPLE_ROWS)
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    df = load_criteo_sample(
        sample_rows=args.sample_rows,
        force_download=args.force_download,
    )

    print(df.head())


if __name__ == "__main__":
    main()