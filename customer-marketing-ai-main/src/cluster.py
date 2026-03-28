from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


INPUT_DATA_PATH = Path("data/processed/marketing_campaign_clean.csv")
OUTPUT_DATA_PATH = Path("data/processed/marketing_campaign_clustered.csv")


NUM_COLS = ["Income", "Age", "Children", "TotalSpending", "TotalPurchases"]
CAT_COLS = ["Education", "Marital_Status"]


def load_data(path: Path = INPUT_DATA_PATH) -> pd.DataFrame:
    """
    Load cleaned dataset.
    """
    if not path.exists():
        raise FileNotFoundError(f"Input dataset not found at: {path}")

    return pd.read_csv(path)


def build_preprocessor() -> ColumnTransformer:
    """
    Build preprocessing pipeline for clustering.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
        ]
    )


def apply_clustering(df: pd.DataFrame, n_clusters: int = 4) -> tuple[pd.DataFrame, KMeans]:
    """
    Apply K-Means clustering and add ClusterID column.
    """
    df = df.copy()

    preprocessor = build_preprocessor()
    x_unsup = preprocessor.fit_transform(df[NUM_COLS + CAT_COLS])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["ClusterID"] = kmeans.fit_predict(x_unsup)

    return df, kmeans


def save_data(df: pd.DataFrame, path: Path = OUTPUT_DATA_PATH) -> None:
    """
    Save clustered dataset.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def describe_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return simple aggregated statistics by cluster.
    """
    summary = (
        df.groupby("ClusterID")[NUM_COLS]
        .mean()
        .round(2)
        .sort_index()
    )
    return summary


def main() -> None:
    df = load_data()
    df_clustered, kmeans = apply_clustering(df, n_clusters=4)
    save_data(df_clustered)

    cluster_summary = describe_clusters(df_clustered)

    print("Clustering completed successfully.")
    print(f"Saved file: {OUTPUT_DATA_PATH}")
    print("\nCluster distribution:")
    print(df_clustered["ClusterID"].value_counts().sort_index())

    print("\nCluster summary (mean values):")
    print(cluster_summary)

    print("\nPreview:")
    print(
        df_clustered[
            ["Income", "Age", "Children", "TotalSpending", "TotalPurchases", "ClusterID"]
        ].head()
    )


if __name__ == "__main__":
    main()