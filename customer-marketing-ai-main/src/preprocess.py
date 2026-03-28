from pathlib import Path
import pandas as pd


RAW_DATA_PATH = Path("data/raw/marketing_campaign.csv")
PROCESSED_DATA_PATH = Path("data/processed/marketing_campaign_clean.csv")


def load_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load raw dataset.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    return pd.read_csv(path, sep="\t")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create simple engineered features.
    """
    df = df.copy()

    df["Age"] = 2026 - df["Year_Birth"]
    df["Children"] = df["Kidhome"] + df["Teenhome"]
    df["TotalSpending"] = (
        df["MntWines"]
        + df["MntFruits"]
        + df["MntMeatProducts"]
        + df["MntFishProducts"]
        + df["MntSweetProducts"]
        + df["MntGoldProds"]
    )
    df["TotalPurchases"] = (
        df["NumWebPurchases"]
        + df["NumCatalogPurchases"]
        + df["NumStorePurchases"]
    )

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply agreed cleaning rules.
    """
    df = df.copy()

    df = df.dropna(subset=["Income", "Response"])

    return df


def save_data(df: pd.DataFrame, path: Path = PROCESSED_DATA_PATH) -> None:
    """
    Save cleaned dataset.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    df = load_data()
    df = build_features(df)
    df = clean_data(df)
    save_data(df)

    print("Preprocessing completed successfully.")
    print(f"Saved file: {PROCESSED_DATA_PATH}")
    print(f"Final shape: {df.shape}")
    print("\nNew columns added:")
    print(["Age", "Children", "TotalSpending", "TotalPurchases"])
    print("\nPreview:")
    print(df[["Income", "Response", "Age", "Children", "TotalSpending", "TotalPurchases"]].head())


if __name__ == "__main__":
    main()