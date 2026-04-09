from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

try:
    from .config import load_params, resolve_path
except ImportError:
    from config import load_params, resolve_path


def load_predictions(path: Path) -> pd.DataFrame:
    """Load prediction results from a CSV file."""
    if not path.exists():
        raise FileNotFoundError(f"Results file not found at: {path}")

    df = pd.read_csv(path)
    required_cols = {"y_true", "y_prob", "y_pred"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in {path.name}: {sorted(missing_cols)}")

    return df


def compute_metrics(df: pd.DataFrame) -> dict:
    """Compute the classification metrics used for the project report."""
    y_true = df["y_true"]
    y_prob = df["y_prob"]
    y_pred = df["y_pred"]

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def compare_models(baseline_metrics: dict, hybrid_metrics: dict) -> dict:
    """Compute the delta hybrid - baseline for the main metrics."""
    comparison = {}
    for metric_name in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
        comparison[metric_name] = round(
            hybrid_metrics[metric_name] - baseline_metrics[metric_name],
            4,
        )

    return comparison


def load_stability_summary(path: Path) -> list[dict] | None:
    """Load optional multi-run stability results if available."""
    if not path.exists():
        return None

    return pd.read_csv(path).to_dict(orient="records")


def save_metrics(summary: dict, path: Path) -> None:
    """Persist the evaluation summary as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def print_metrics(title: str, metrics: dict) -> None:
    """Pretty print a metric block in the terminal."""
    print(f"\n=== {title} ===")
    print(f"Accuracy:  {metrics['accuracy']}")
    print(f"Precision: {metrics['precision']}")
    print(f"Recall:    {metrics['recall']}")
    print(f"F1-score:  {metrics['f1_score']}")
    print(f"ROC-AUC:   {metrics['roc_auc']}")
    print(f"Confusion Matrix: {metrics['confusion_matrix']}")


def main() -> dict:
    config = load_params()
    paths = config["paths"]

    baseline_results_path = resolve_path(paths["baseline_results"])
    hybrid_results_path = resolve_path(paths["hybrid_results"])
    metrics_output_path = resolve_path(paths["metrics_summary"])
    stability_summary_path = resolve_path(paths["stability_summary"])

    baseline_df = load_predictions(baseline_results_path)
    hybrid_df = load_predictions(hybrid_results_path)

    baseline_metrics = compute_metrics(baseline_df)
    hybrid_metrics = compute_metrics(hybrid_df)
    comparison = compare_models(baseline_metrics, hybrid_metrics)
    stability_summary = load_stability_summary(stability_summary_path)

    summary = {
        "baseline": baseline_metrics,
        "hybrid": hybrid_metrics,
        "hybrid_minus_baseline": comparison,
    }
    if stability_summary is not None:
        summary["stability_summary"] = stability_summary

    print_metrics("Baseline Model", baseline_metrics)
    print_metrics("Hybrid Model", hybrid_metrics)

    print("\n=== Improvement: Hybrid - Baseline ===")
    for metric_name, value in comparison.items():
        print(f"{metric_name}: {value}")

    if stability_summary is not None:
        print("\n=== Stability Summary ===")
        print(pd.DataFrame(stability_summary))

    save_metrics(summary, metrics_output_path)
    print(f"\nMetrics summary saved to: {metrics_output_path}")

    return summary


if __name__ == "__main__":
    main()
