from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from .config import load_params, resolve_path
except ImportError:
    from config import load_params, resolve_path


def load_data(path: Path) -> pd.DataFrame:
    """Load the raw marketing dataset from a tab-separated CSV file."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    return pd.read_csv(path, sep="\t")


def build_features(df: pd.DataFrame, current_year: int) -> pd.DataFrame:
    """Create the engineered features used across the project."""
    df = df.copy()

    spending_cols = [
        "MntWines",
        "MntFruits",
        "MntMeatProducts",
        "MntFishProducts",
        "MntSweetProducts",
        "MntGoldProds",
    ]
    purchase_cols = ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]

    df["Age"] = current_year - df["Year_Birth"]
    df["Children"] = df["Kidhome"].fillna(0) + df["Teenhome"].fillna(0)
    df["IsParent"] = (df["Children"] > 0).astype(int)
    df["TotalSpending"] = df[spending_cols].sum(axis=1)
    df["TotalPurchases"] = df[purchase_cols].sum(axis=1)

    return df


def clean_data(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    """Apply the shared cleaning rules used in the notebook and scripts."""
    df = df.copy()
    preprocessing = config["preprocessing"]
    current_year = preprocessing["current_year"]

    rows_before = len(df)
    df.columns = df.columns.str.strip()
    df["Income"] = pd.to_numeric(df["Income"], errors="coerce")
    df["Response"] = pd.to_numeric(df["Response"], errors="coerce")

    for col in ["Education", "Marital_Status"]:
        df[col] = df[col].astype("string").str.strip()

    df = df.dropna(subset=["Income", "Response"]).copy()
    df["Response"] = df["Response"].astype(int)
    rows_after_missing_drop = len(df)

    df = build_features(df, current_year=current_year)

    invalid_age_mask = ~df["Age"].between(
        preprocessing["min_age"],
        preprocessing["max_age"],
    )
    invalid_age_rows = int(invalid_age_mask.sum())
    df = df.loc[~invalid_age_mask].copy()

    if not pd.api.types.is_numeric_dtype(df["Income"]):
        raise TypeError("Income must be numeric after cleaning.")

    response_values = set(df["Response"].dropna().unique().tolist())
    if not response_values.issubset({0, 1}):
        raise ValueError(f"Unexpected Response values: {sorted(response_values)}")

    lower_q = df["Income"].quantile(preprocessing["income_lower_quantile"])
    upper_q = df["Income"].quantile(preprocessing["income_upper_quantile"])
    rows_before_outlier_filter = len(df)
    df = df[df["Income"].between(lower_q, upper_q)].copy()
    rows_after_outlier_filter = len(df)

    summary = {
        "rows_before_cleaning": rows_before,
        "rows_after_missing_drop": rows_after_missing_drop,
        "invalid_age_rows_removed": invalid_age_rows,
        "rows_removed_income_outliers": rows_before_outlier_filter - rows_after_outlier_filter,
        "income_lower_quantile_value": round(float(lower_q), 2),
        "income_upper_quantile_value": round(float(upper_q), 2),
        "final_rows": len(df),
        "final_columns": len(df.columns),
    }

    return df, summary


def save_data(df: pd.DataFrame, path: Path) -> None:
    """Save the cleaned dataset to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> pd.DataFrame:
    config = load_params()
    paths = config["paths"]

    raw_data_path = resolve_path(paths["raw_data"])
    processed_data_path = resolve_path(paths["processed_data"])

    df = load_data(raw_data_path)
    cleaned_df, summary = clean_data(df, config)
    save_data(cleaned_df, processed_data_path)

    print("Preprocessing completed successfully.")
    print(f"Saved file: {processed_data_path}")
    print(f"Final shape: {cleaned_df.shape}")
    print("Summary:")
    for key, value in summary.items():
        print(f" - {key}: {value}")

    preview_cols = [
        "Income",
        "Response",
        "Age",
        "Children",
        "IsParent",
        "TotalSpending",
        "TotalPurchases",
    ]
    print("\nPreview:")
    print(cleaned_df[preview_cols].head())

    return cleaned_df


if __name__ == "__main__":
    main()
