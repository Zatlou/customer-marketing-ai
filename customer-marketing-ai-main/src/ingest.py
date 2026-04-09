from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from .config import load_params, resolve_path
except ImportError:
    from config import load_params, resolve_path


def load_data(path: Path) -> pd.DataFrame:
    """Load the raw marketing dataset and perform very light validation."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    df = pd.read_csv(path, sep="\t")
    df.columns = df.columns.str.strip()

    required_cols = {
        "Year_Birth",
        "Income",
        "Education",
        "Marital_Status",
        "Kidhome",
        "Teenhome",
        "Response",
    }
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in the raw dataset: {sorted(missing_cols)}")

    return df


def main() -> pd.DataFrame:
    config = load_params()
    raw_data_path = resolve_path(config["paths"]["raw_data"])
    df = load_data(raw_data_path)

    print("Dataset loaded successfully.")
    print(f"Path: {raw_data_path}")
    print(f"Shape: {df.shape}")
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nFirst 5 rows:")
    print(df.head())

    return df


if __name__ == "__main__":
    main()
