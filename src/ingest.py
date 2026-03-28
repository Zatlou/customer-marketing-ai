from pathlib import Path
import pandas as pd


DATA_PATH = Path("data/raw/marketing_campaign.csv")


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Load the raw marketing dataset from a tab-separated CSV file.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    df = pd.read_csv(path, sep="\t")
    return df


def main() -> None:
    df = load_data()

    print("Dataset loaded successfully.")
    print(f"Shape: {df.shape}")
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nFirst 5 rows:")
    print(df.head())


if __name__ == "__main__":
    main()