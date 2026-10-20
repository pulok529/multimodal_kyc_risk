"""Training pipeline and ablation experiments for multimodal KYC risk detection."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader

from kyc_risk.dataset import KYCDataset, fit_scaler, transform_tabular
from kyc_risk.models import FusionModel, ImageOnlyModel, TabularOnlyModel
from kyc_risk.synth import generate, render_document

matplotlib.use("Agg")


def _set_seeds(seed: int) -> None:
    """Set random seeds across standard library, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find threshold that achieves >= 90% precision with max recall, or best F1 fallback."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    if len(thresholds) == 0:
        return 0.50

    # Look for candidates with precision >= 0.90
    candidates = [
        (i, float(thresholds[i]), float(recalls[i]), float(precisions[i]))
        for i in range(len(thresholds))
        if precisions[i] >= 0.90
    ]
    if candidates:
        # Highest recall among those with precision >= 0.90
        best = max(candidates, key=lambda x: (x[2], x[3]))
        return float(best[1])

    # Fallback to threshold that maximizes F1 score
    f1s = [
        (2.0 * p * r) / (p + r) if (p + r) > 0 else 0.0
        for p, r in zip(precisions[:-1], recalls[:-1], strict=False)
    ]
    best_idx = int(np.argmax(f1s)) if f1s else 0
    return float(thresholds[best_idx])


def _train_single_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    pos_weight: torch.Tensor,
    epochs: int = 8,
    lr: float = 1e-3,
    patience: int = 3,
) -> tuple[nn.Module, float]:
    """Train a model with BCEWithLogitsLoss, pos_weight, and early stopping on val PR-AUC."""
    model = model.to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_pr_auc = -1.0
    best_weights = copy.deepcopy(model.state_dict())
    patience_counter = 0

    for _epoch in range(1, epochs + 1):
        model.train()
        for x_img, x_tab, y in train_loader:
            x_img, x_tab, y = x_img.to(device), x_tab.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x_img, x_tab)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

        # Evaluate on validation set
        val_probs, val_trues = _predict_dataset(model, val_loader, device)
        val_pr_auc = float(average_precision_score(val_trues, val_probs))

        if val_pr_auc > best_pr_auc + 1e-4:
            best_pr_auc = val_pr_auc
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_weights)
    return model, best_pr_auc


@torch.no_grad()
def _predict_dataset(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Compute predicted probabilities and extract ground truth labels."""
    model.eval()
    all_probs: list[float] = []
    all_trues: list[float] = []

    for x_img, x_tab, y in loader:
        x_img, x_tab = x_img.to(device), x_tab.to(device)
        logits = model(x_img, x_tab)
        probs = torch.sigmoid(logits).cpu().numpy().tolist()
        all_probs.extend(probs)
        all_trues.extend(y.numpy().tolist())

    return np.array(all_probs, dtype=np.float32), np.array(all_trues, dtype=np.float32)


def _generate_samples_grid(out_path: Path, docs_path: Path | None = None) -> None:
    """Render and save an 8x2 grid of synthetic genuine vs tampered ID documents."""
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(2, 8, figsize=(14, 4), facecolor="#0B1020")
    fig.subplots_adjust(wspace=0.1, hspace=0.3)

    for col in range(8):
        gen_img = render_document(rng, tampered=False, size=96)
        axes[0, col].imshow(gen_img, cmap="gray", vmin=0, vmax=255)
        axes[0, col].axis("off")
        if col == 0:
            axes[0, col].set_title(
                "Genuine", loc="left", color="#22D3EE", fontsize=11, fontweight="bold"
            )

        tam_img = render_document(rng, tampered=True, size=96)
        axes[1, col].imshow(tam_img, cmap="gray", vmin=0, vmax=255)
        axes[1, col].axis("off")
        if col == 0:
            axes[1, col].set_title(
                "Tampered (Altered)", loc="left", color="#F43F5E", fontsize=11, fontweight="bold"
            )

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="#0B1020")
    if docs_path is not None:
        docs_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(docs_path, dpi=200, bbox_inches="tight", facecolor="#0B1020")
    plt.close(fig)


def _generate_ablation_chart(
    metrics: dict[str, Any], out_path: Path, docs_path: Path | None = None
) -> None:
    """Plot and save a dark-themed ablation bar chart comparing model PR-AUC scores."""
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="#0B1020")
    ax.set_facecolor("#0B1020")

    labels = [
        "Fusion\n(Image + Tabular)",
        "Image Only\n(CNN Baseline)",
        "Tabular Only\n(MLP Baseline)",
    ]
    keys = ["fusion", "image_only", "tabular_only"]
    values = [metrics[k]["pr_auc"] for k in keys]
    colors = ["#22D3EE", "#8B5CF6", "#94A3B8"]

    bars = ax.bar(labels, values, color=colors, width=0.55, edgecolor="#1E293B", linewidth=1.5)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("PR-AUC (Precision-Recall Area)", color="#F8FAFC", fontsize=11, labelpad=10)
    ax.set_title(
        "KYC Fraud Detection Ablation: Multimodal vs Single Modality",
        color="#F8FAFC",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )

    ax.tick_params(colors="#94A3B8", labelsize=10)
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.grid(axis="y", color="#1E293B", linestyle="--", alpha=0.7)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color="#F8FAFC",
            fontsize=11,
            fontweight="bold",
        )

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="#0B1020")
    if docs_path is not None:
        docs_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(docs_path, dpi=200, bbox_inches="tight", facecolor="#0B1020")
    plt.close(fig)


def train_pipeline(
    n: int = 6000,
    epochs: int = 8,
    out_dir: str = "artifacts",
    seed: int = 42,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> dict[str, Any]:
    """Execute full data generation, multimodal training, ablation, and artifact export.

    Args:
        n: Total synthetic applicants to simulate.
        epochs: Max training epochs per model.
        out_dir: Destination folder for exported model and evaluation artifacts.
        seed: Random seed for data generation and model parameter initialization.
        batch_size: Mini-batch size.
        lr: AdamW learning rate.

    Returns:
        Dictionary of final test evaluation metrics across all three ablation models.
    """
    _set_seeds(seed)
    num_threads = min(4, os.cpu_count() or 1)
    torch.set_num_threads(num_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Generate synthetic dataset
    images, tabular_df, labels = generate(n=n, seed=seed)

    # 2. Stratified train/val/test split (70 / 15 / 15)
    train_idx, temp_idx = train_test_split(
        np.arange(n), test_size=0.30, random_state=seed, stratify=labels
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=seed, stratify=labels[temp_idx]
    )

    # 3. Fit scaler strictly on training tabular split
    train_tabular_df = tabular_df.iloc[train_idx].reset_index(drop=True)
    val_tabular_df = tabular_df.iloc[val_idx].reset_index(drop=True)
    test_tabular_df = tabular_df.iloc[test_idx].reset_index(drop=True)

    scaler = fit_scaler(train_tabular_df)
    with open(out_path / "scaler.json", "w", encoding="utf-8") as f:
        json.dump(scaler, f, indent=2)

    # Pre-standardise tabular matrices
    train_tab = transform_tabular(train_tabular_df, scaler)
    val_tab = transform_tabular(val_tabular_df, scaler)
    test_tab = transform_tabular(test_tabular_df, scaler)

    # Build DataLoaders
    train_ds = KYCDataset(images[train_idx], train_tab, labels[train_idx])
    val_ds = KYCDataset(images[val_idx], val_tab, labels[val_idx])
    test_ds = KYCDataset(images[test_idx], test_tab, labels[test_idx])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Class balance pos_weight for BCEWithLogitsLoss
    n_pos = float(np.sum(labels[train_idx]))
    n_neg = float(len(train_idx) - n_pos)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float32, device=device)

    # 4. Train three models: Fusion, Image-Only, Tabular-Only
    models_to_train: dict[str, nn.Module] = {
        "fusion": FusionModel(tabular_dim=train_tab.shape[1]),
        "image_only": ImageOnlyModel(),
        "tabular_only": TabularOnlyModel(tabular_dim=train_tab.shape[1]),
    }

    metrics: dict[str, Any] = {}

    for name, model_instance in models_to_train.items():
        trained_model, _ = _train_single_model(
            model=model_instance,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            pos_weight=pos_weight,
            epochs=epochs,
            lr=lr,
            patience=3,
        )

        # Tune threshold on validation set
        val_probs, val_trues = _predict_dataset(trained_model, val_loader, device)
        val_thresh = _find_best_threshold(val_trues, val_probs)

        # Final evaluation on holdout test set
        test_probs, test_trues = _predict_dataset(trained_model, test_loader, device)
        test_preds = (test_probs >= val_thresh).astype(int)

        roc_auc = float(roc_auc_score(test_trues, test_probs))
        pr_auc = float(average_precision_score(test_trues, test_probs))
        rec = float(recall_score(test_trues, test_preds, zero_division=0))
        prec = float(precision_score(test_trues, test_preds, zero_division=0))
        f1 = float(f1_score(test_trues, test_preds, zero_division=0))

        metrics[name] = {
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "recall": round(rec, 4),
            "precision": round(prec, 4),
            "f1": round(f1, 4),
            "val_threshold": round(val_thresh, 4),
        }

        # Save fusion model state_dict as primary deployment checkpoint
        if name == "fusion":
            torch.save(trained_model.state_dict(), out_path / "model.pt")

    # 5. Export metrics.json
    with open(out_path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # 6. Generate visualizations for artifacts/ and docs/
    docs_dir = Path("docs")
    _generate_samples_grid(out_path / "samples.png", docs_dir / "samples.png")
    _generate_ablation_chart(metrics, out_path / "ablation.png", docs_dir / "ablation.png")

    # 7. Print formatted markdown table (all lines <= 100 characters)
    table_lines = [
        "| Model | PR-AUC | ROC-AUC | Recall@ValThresh | Precision | F1 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for key in ("fusion", "image_only", "tabular_only"):
        m = metrics[key]
        table_lines.append(
            f"| {key:<12} | {m['pr_auc']:.4f} | {m['roc_auc']:.4f} | "
            f"{m['recall']:.4f} | {m['precision']:.4f} | {m['f1']:.4f} |"
        )
    print("\n" + "\n".join(table_lines) + "\n")

    return metrics


def main() -> None:
    """CLI entrypoint for training pipeline."""
    parser = argparse.ArgumentParser(
        description="Train multimodal KYC risk models with ablation benchmarks."
    )
    parser.add_argument("--n", type=int, default=6000, help="Number of synthetic samples.")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs.")
    parser.add_argument("--out", type=str, default="artifacts", help="Artifacts output directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")

    args = parser.parse_args()
    train_pipeline(
        n=args.n,
        epochs=args.epochs,
        out_dir=args.out,
        seed=args.seed,
        batch_size=args.batch_size,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
