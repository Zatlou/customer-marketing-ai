from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from .config import load_params, resolve_path
except ImportError:
    from config import load_params, resolve_path


def load_data(path: Path) -> pd.DataFrame:
    """Load the cleaned dataset ready for clustering."""
    if not path.exists():
        raise FileNotFoundError(f"Input dataset not found at: {path}")

    return pd.read_csv(path)


def build_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    """Create the preprocessing block used by K-Means."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )


def search_best_k(
    x_prepared,
    k_min: int,
    k_max: int,
    random_state: int,
    n_init: int,
) -> tuple[pd.DataFrame, int]:
    """Evaluate a range of cluster counts with silhouette score."""
    records = []

    for k in range(k_min, k_max + 1):
        model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        labels = model.fit_predict(x_prepared)
        score = silhouette_score(x_prepared, labels)
        records.append({"k": k, "silhouette_score": round(float(score), 6)})

    silhouette_df = pd.DataFrame(records)
    best_k = int(silhouette_df.loc[silhouette_df["silhouette_score"].idxmax(), "k"])

    return silhouette_df, best_k


def add_cluster_interpretation(cluster_summary: pd.DataFrame) -> pd.DataFrame:
    """Generate a short business-oriented interpretation per cluster."""
    cluster_summary = cluster_summary.copy()

    spend_q1, spend_q3 = cluster_summary["TotalSpending_mean"].quantile([0.25, 0.75])
    income_q1, income_q3 = cluster_summary["Income_mean"].quantile([0.25, 0.75])
    response_q1, response_q3 = cluster_summary["ResponseRate"].quantile([0.25, 0.75])

    def interpret(row: pd.Series) -> str:
        if row["TotalSpending_mean"] >= spend_q3:
            spending_desc = "high spending"
        elif row["TotalSpending_mean"] <= spend_q1:
            spending_desc = "lower spending"
        else:
            spending_desc = "mid spending"

        if row["Income_mean"] >= income_q3:
            income_desc = "higher income"
        elif row["Income_mean"] <= income_q1:
            income_desc = "more modest income"
        else:
            income_desc = "mid income"

        if row["ResponseRate"] >= response_q3:
            response_desc = "more responsive"
        elif row["ResponseRate"] <= response_q1:
            response_desc = "less responsive"
        else:
            response_desc = "average response"

        return f"{spending_desc}, {income_desc}, {response_desc}"

    cluster_summary["Interpretation"] = cluster_summary.apply(interpret, axis=1)
    cluster_summary["ResponseRatePct"] = (100 * cluster_summary["ResponseRate"]).round(2)

    return cluster_summary


def build_cluster_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the main business indicators by cluster."""
    summary = (
        df.groupby("ClusterID")
        .agg(
            Count=("ClusterID", "size"),
            Income_mean=("Income", "mean"),
            Age_mean=("Age", "mean"),
            Children_mean=("Children", "mean"),
            IsParent_rate=("IsParent", "mean"),
            TotalSpending_mean=("TotalSpending", "mean"),
            TotalPurchases_mean=("TotalPurchases", "mean"),
            ResponseRate=("Response", "mean"),
        )
        .round(2)
        .reset_index()
    )

    return add_cluster_interpretation(summary)


def save_cluster_pca_plot(x_prepared, labels, output_path: Path) -> None:
    """Create a simple 2D PCA view of the learned clusters."""
    projection = PCA(n_components=2).fit_transform(x_prepared)
    plot_df = pd.DataFrame(
        {
            "PCA1": projection[:, 0],
            "PCA2": projection[:, 1],
            "ClusterID": labels,
        }
    )

    plt.figure(figsize=(8, 6))
    for cluster_id, group_df in plot_df.groupby("ClusterID"):
        plt.scatter(group_df["PCA1"], group_df["PCA2"], label=f"Cluster {cluster_id}", alpha=0.7, s=35)
    plt.title("Cluster projection with PCA")
    plt.xlabel("PCA component 1")
    plt.ylabel("PCA component 2")
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> pd.DataFrame:
    config = load_params()
    paths = config["paths"]
    clustering = config["clustering"]
    features = config["features"]

    input_data_path = resolve_path(paths["processed_data"])
    output_data_path = resolve_path(paths["clustered_data"])
    silhouette_output_path = resolve_path(paths["silhouette_scores"])
    cluster_summary_output_path = resolve_path(paths["cluster_summary"])
    cluster_pca_output_path = resolve_path(paths["cluster_pca_plot"])

    df = load_data(input_data_path)
    num_cols = features["cluster_numeric"]
    cat_cols = features["categorical"]

    preprocessor = build_preprocessor(num_cols, cat_cols)
    x_prepared = preprocessor.fit_transform(df[num_cols + cat_cols])
    x_prepared = x_prepared.toarray() if hasattr(x_prepared, "toarray") else x_prepared

    silhouette_df, best_k = search_best_k(
        x_prepared=x_prepared,
        k_min=clustering["k_min"],
        k_max=clustering["k_max"],
        random_state=clustering["random_state"],
        n_init=clustering["n_init"],
    )

    final_model = KMeans(
        n_clusters=best_k,
        random_state=clustering["random_state"],
        n_init=clustering["n_init"],
    )

    clustered_df = df.copy()
    clustered_df["ClusterID"] = final_model.fit_predict(x_prepared)
    cluster_summary_df = build_cluster_summary(clustered_df)

    output_data_path.parent.mkdir(parents=True, exist_ok=True)
    silhouette_output_path.parent.mkdir(parents=True, exist_ok=True)
    cluster_summary_output_path.parent.mkdir(parents=True, exist_ok=True)

    clustered_df.to_csv(output_data_path, index=False)
    silhouette_df.to_csv(silhouette_output_path, index=False)
    cluster_summary_df.to_csv(cluster_summary_output_path, index=False)
    save_cluster_pca_plot(
        x_prepared=x_prepared,
        labels=clustered_df["ClusterID"].to_numpy(),
        output_path=cluster_pca_output_path,
    )

    print("Clustering completed successfully.")
    print(f"Saved clustered dataset: {output_data_path}")
    print(f"Saved silhouette scores: {silhouette_output_path}")
    print(f"Saved cluster summary: {cluster_summary_output_path}")
    print(f"Saved PCA visualization: {cluster_pca_output_path}")
    print(f"Selected best k with silhouette score: {best_k}")
    print("\nSilhouette scores:")
    print(silhouette_df)
    print("\nCluster summary:")
    print(cluster_summary_df)

    return clustered_df


if __name__ == "__main__":
    main()
