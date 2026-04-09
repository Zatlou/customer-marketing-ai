from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

try:
    from .config import load_params, resolve_path
except ImportError:
    from config import load_params, resolve_path


class MarketingDataset(Dataset):
    """Tiny dataset wrapper for the PyTorch classifier."""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class ResponseMLP(nn.Module):
    """Simple MLP used for the supervised part of the course project."""

    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float):
        super().__init__()

        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def set_seed(seed: int) -> None:
    """Set all major seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(path: Path) -> pd.DataFrame:
    """Load the clustered dataset used for supervised training."""
    if not path.exists():
        raise FileNotFoundError(f"Clustered dataset not found at: {path}")

    return pd.read_csv(path)


def prepare_dataloaders(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    test_size: float,
    split_seed: int,
    batch_size_train: int,
    batch_size_test: int,
) -> tuple[DataLoader, DataLoader, pd.Series, pd.Series, np.ndarray, np.ndarray]:
    """Split, scale, and package the data without train/test leakage."""
    if int(df[feature_cols + [target_col]].isna().sum().sum()) != 0:
        raise ValueError("Missing values remain in the training data.")

    x_train_df, x_test_df, y_train, y_test = train_test_split(
        df[feature_cols],
        df[target_col],
        test_size=test_size,
        stratify=df[target_col],
        random_state=split_seed,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train_df)
    x_test = scaler.transform(x_test_df)

    train_loader = DataLoader(
        MarketingDataset(x_train, y_train.to_numpy()),
        batch_size=batch_size_train,
        shuffle=True,
    )
    test_loader = DataLoader(
        MarketingDataset(x_test, y_test.to_numpy()),
        batch_size=batch_size_test,
        shuffle=False,
    )

    return train_loader, test_loader, y_train, y_test, x_train_df.index.to_numpy(), x_test_df.index.to_numpy()


def train_model(
    train_loader: DataLoader,
    input_dim: int,
    hidden_dims: list[int],
    dropout: float,
    learning_rate: float,
    epochs: int,
    pos_weight_value: float,
) -> ResponseMLP:
    """Train the PyTorch MLP on the training split."""
    model = ResponseMLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for _ in range(epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

    return model


def predict_model(model: ResponseMLP, test_loader: DataLoader, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Generate class predictions and probabilities on the test split."""
    model.eval()
    probabilities: list[float] = []

    with torch.no_grad():
        for x_batch, _ in test_loader:
            batch_probabilities = torch.sigmoid(model(x_batch)).squeeze().cpu().numpy()
            batch_probabilities = np.atleast_1d(batch_probabilities)
            probabilities.extend(batch_probabilities.tolist())

    y_prob = np.array(probabilities)
    y_pred = (y_prob >= threshold).astype(int)

    return y_pred, y_prob


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute the main evaluation metrics used in the project."""
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
    }


def run_experiment(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    settings: dict,
    split_seed: int,
    model_seed: int,
) -> tuple[pd.DataFrame, dict]:
    """Train and evaluate one baseline or hybrid experiment."""
    set_seed(model_seed)

    train_loader, test_loader, y_train, y_test, _, test_indices = prepare_dataloaders(
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        test_size=settings["test_size"],
        split_seed=split_seed,
        batch_size_train=settings["batch_size_train"],
        batch_size_test=settings["batch_size_test"],
    )

    positive_count = max(int(y_train.sum()), 1)
    negative_count = max(int(len(y_train) - positive_count), 1)
    pos_weight_value = negative_count / positive_count

    model = train_model(
        train_loader=train_loader,
        input_dim=len(feature_cols),
        hidden_dims=settings["hidden_dims"],
        dropout=settings["dropout"],
        learning_rate=settings["learning_rate"],
        epochs=settings["epochs"],
        pos_weight_value=pos_weight_value,
    )

    y_pred, y_prob = predict_model(
        model=model,
        test_loader=test_loader,
        threshold=settings["classification_threshold"],
    )

    results_df = pd.DataFrame(
        {
            "ID": df.loc[test_indices, "ID"].values,
            "y_true": y_test.to_numpy(),
            "y_pred": y_pred,
            "y_prob": y_prob,
        }
    )

    metrics = compute_metrics(y_true=y_test, y_pred=y_pred, y_prob=y_prob)

    return results_df, metrics


def main() -> dict:
    config = load_params()
    paths = config["paths"]
    training = config["training"]
    features = config["features"]

    clustered_data_path = resolve_path(paths["clustered_data"])
    baseline_results_path = resolve_path(paths["baseline_results"])
    hybrid_results_path = resolve_path(paths["hybrid_results"])
    stability_summary_path = resolve_path(paths["stability_summary"])

    df = load_data(clustered_data_path)
    target_col = "Response"
    baseline_features = features["baseline_model"]
    hybrid_features = features["hybrid_model"]

    baseline_results_df, baseline_metrics = run_experiment(
        df=df,
        feature_cols=baseline_features,
        target_col=target_col,
        settings=training,
        split_seed=training["random_state"],
        model_seed=training["random_state"],
    )
    hybrid_results_df, hybrid_metrics = run_experiment(
        df=df,
        feature_cols=hybrid_features,
        target_col=target_col,
        settings=training,
        split_seed=training["random_state"],
        model_seed=training["random_state"],
    )

    baseline_results_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_results_df.to_csv(baseline_results_path, index=False)
    hybrid_results_df.to_csv(hybrid_results_path, index=False)

    stability_records = []
    for seed in training["stability_seeds"]:
        for model_name, feature_cols in [
            ("baseline", baseline_features),
            ("hybrid", hybrid_features),
        ]:
            _, metrics = run_experiment(
                df=df,
                feature_cols=feature_cols,
                target_col=target_col,
                settings=training,
                split_seed=seed,
                model_seed=seed,
            )
            stability_records.append({"model": model_name, "seed": seed, **metrics})

    stability_df = pd.DataFrame(stability_records)
    stability_summary = (
        stability_df.groupby("model")[["accuracy", "precision", "recall", "f1_score", "roc_auc"]]
        .agg(["mean", "std"])
        .round(4)
    )
    stability_summary.columns = ["_".join(col).strip("_") for col in stability_summary.columns]
    stability_summary = stability_summary.reset_index()
    stability_summary_path.parent.mkdir(parents=True, exist_ok=True)
    stability_summary.to_csv(stability_summary_path, index=False)

    print("Training completed successfully.")
    print(f"Saved baseline results: {baseline_results_path}")
    print(f"Saved hybrid results: {hybrid_results_path}")
    print(f"Saved stability summary: {stability_summary_path}")
    print("\nBaseline metrics:")
    print(baseline_metrics)
    print("\nHybrid metrics:")
    print(hybrid_metrics)
    print("\nStability summary:")
    print(stability_summary)

    return {
        "baseline": baseline_metrics,
        "hybrid": hybrid_metrics,
    }


if __name__ == "__main__":
    main()
