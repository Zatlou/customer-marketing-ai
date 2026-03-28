from pathlib import Path
import json

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


BASE_DIR = Path("reports")
BASELINE_RESULTS_PATH = BASE_DIR / "baseline_results.csv"
HYBRID_RESULTS_PATH = BASE_DIR / "hybrid_results.csv"
METRICS_OUTPUT_PATH = BASE_DIR / "metrics_summary.json"


def load_predictions(path: Path) -> pd.DataFrame:
    """
    Load prediction results from a CSV file.

    Expected columns:
    - y_true
    - y_prob
    - y_pred
    """
    if not path.exists():
        raise FileNotFoundError(f"Results file not found at: {path}")

    df = pd.read_csv(path)

    required_cols = {"y_true", "y_prob", "y_pred"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Missing required columns in {path.name}: {sorted(missing_cols)}"
        )

    return df


def compute_metrics(df: pd.DataFrame) -> dict:
    """
    Compute classification metrics from predictions.
    """
    y_true = df["y_true"]
    y_prob = df["y_prob"]
    y_pred = df["y_pred"]

    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    return metrics


def compare_models(baseline_metrics: dict, hybrid_metrics: dict) -> dict:
    """
    Compare baseline and hybrid metrics.
    """
    comparison = {}

    for metric_name in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
        comparison[metric_name] = round(
            hybrid_metrics[metric_name] - baseline_metrics[metric_name],
            4,
        )

    return comparison


def save_metrics(summary: dict, path: Path = METRICS_OUTPUT_PATH) -> None:
    """
    Save evaluation summary as JSON.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def print_metrics(title: str, metrics: dict) -> None:
    """
    Pretty print metrics in terminal.
    """
    print(f"\n=== {title} ===")
    print(f"Accuracy:  {metrics['accuracy']}")
    print(f"Precision: {metrics['precision']}")
    print(f"Recall:    {metrics['recall']}")
    print(f"F1-score:  {metrics['f1_score']}")
    print(f"ROC-AUC:   {metrics['roc_auc']}")
    print(f"Confusion Matrix: {metrics['confusion_matrix']}")


def main() -> None:
    baseline_df = load_predictions(BASELINE_RESULTS_PATH)
    hybrid_df = load_predictions(HYBRID_RESULTS_PATH)

    baseline_metrics = compute_metrics(baseline_df)
    hybrid_metrics = compute_metrics(hybrid_df)
    comparison = compare_models(baseline_metrics, hybrid_metrics)

    summary = {
        "baseline": baseline_metrics,
        "hybrid": hybrid_metrics,
        "hybrid_minus_baseline": comparison,
    }

    print_metrics("Baseline Model", baseline_metrics)
    print_metrics("Hybrid Model", hybrid_metrics)

    print("\n=== Improvement: Hybrid - Baseline ===")
    for metric_name, value in comparison.items():
        print(f"{metric_name}: {value}")

    save_metrics(summary)

    print(f"\nMetrics summary saved to: {METRICS_OUTPUT_PATH}")


if __name__ == "__main__":
    main()