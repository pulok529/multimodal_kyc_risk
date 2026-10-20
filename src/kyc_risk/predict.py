"""Inference and risk evaluation functions for multimodal KYC applicant scoring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from kyc_risk.dataset import transform_tabular
from kyc_risk.models import FusionModel


def load(artifacts_dir: str | Path = "artifacts") -> dict[str, Any]:
    """Load model weights, feature scaler, and validation threshold from artifacts directory.

    Args:
        artifacts_dir: Path to directory containing model.pt, scaler.json, metrics.json.

    Returns:
        Dictionary containing model, scaler, threshold, and metrics.
    """
    path = Path(artifacts_dir)
    model_path = path / "model.pt"
    scaler_path = path / "scaler.json"
    metrics_path = path / "metrics.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler parameters not found at {scaler_path}")

    with open(scaler_path, encoding="utf-8") as f:
        scaler = json.load(f)

    metrics: dict[str, Any] = {}
    threshold = 0.50
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)
            threshold = float(metrics.get("fusion", {}).get("val_threshold", 0.50))

    model = FusionModel(tabular_dim=len(scaler.get("columns", [])) or 9)
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    return {
        "model": model,
        "scaler": scaler,
        "threshold": threshold,
        "metrics": metrics,
    }


def _extract_signals(tabular: dict[str, Any]) -> list[str]:
    """Generate human-readable risk signals from raw tabular attributes."""
    signals: list[str] = []

    selfie_match = float(tabular.get("selfie_match_score", 1.0))
    if selfie_match < 0.60:
        signals.append("Low selfie match score (< 0.60)")

    device_risk = float(tabular.get("device_risk", 0.0))
    if device_risk > 0.70:
        signals.append("High device risk score (> 0.70)")

    if int(tabular.get("pep_flag", 0)) == 1:
        signals.append("Politically Exposed Person (PEP) flag active")

    if int(tabular.get("address_mismatch", 0)) == 1:
        signals.append("Applicant address mismatch detected")

    failed_logins = int(tabular.get("num_failed_logins_30d", 0))
    if failed_logins >= 3:
        signals.append("Repeated failed logins in last 30 days (>= 3)")

    velocity = int(tabular.get("velocity_signups_same_device", 1))
    if velocity > 3:
        signals.append("Elevated signup velocity on device (> 3 accounts)")

    country_risk = float(tabular.get("country_risk", 0.0))
    if country_risk > 0.70:
        signals.append("High country risk score (> 0.70)")

    return signals


def _prepare_image_tensor(image: np.ndarray | Image.Image) -> torch.Tensor:
    """Normalize and format arbitrary document image input to a (1, 1, 96, 96) tensor."""
    if isinstance(image, Image.Image):
        pil_img = image.convert("L").resize((96, 96), resample=Image.Resampling.BILINEAR)
        arr = np.asarray(pil_img, dtype=np.float32) / 255.0
    elif isinstance(image, np.ndarray):
        arr = image.astype(np.float32)
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = arr[0]
        elif arr.ndim == 3 and arr.shape[-1] in (1, 3):
            arr = arr[..., 0]

        if arr.shape != (96, 96):
            pil_img = Image.fromarray(
                np.clip(arr if arr.max() > 1.0 else arr * 255.0, 0, 255).astype(np.uint8),
                mode="L",
            ).resize((96, 96), resample=Image.Resampling.BILINEAR)
            arr = np.asarray(pil_img, dtype=np.float32) / 255.0
        elif arr.max() > 1.0:
            arr = arr / 255.0
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


def predict(
    image: np.ndarray | Image.Image,
    tabular: dict[str, Any],
    artifacts: dict[str, Any] | None = None,
    artifacts_dir: str | Path = "artifacts",
) -> dict[str, Any]:
    """Score an applicant using fused document image and tabular indicators.

    Args:
        image: 96x96 grayscale image as NumPy array or PIL Image.
        tabular: Dictionary of applicant tabular risk features.
        artifacts: Preloaded artifact dictionary (loads from artifacts_dir if None).
        artifacts_dir: Directory to load model and scaler if artifacts is None.

    Returns:
        Dictionary containing risk_probability, decision, threshold, and signals.
    """
    if artifacts is None:
        artifacts = load(artifacts_dir)

    model: FusionModel = artifacts["model"]
    scaler: dict[str, Any] = artifacts["scaler"]
    threshold: float = float(artifacts.get("threshold", 0.50))

    img_tensor = _prepare_image_tensor(image)
    tab_matrix = transform_tabular(tabular, scaler)
    tab_tensor = torch.from_numpy(tab_matrix)

    with torch.no_grad():
        logits = model(img_tensor, tab_tensor)
        risk_probability = float(torch.sigmoid(logits).item())

    decision = "manual_review" if risk_probability >= threshold else "auto_approve"
    signals = _extract_signals(tabular)

    return {
        "risk_probability": round(risk_probability, 4),
        "decision": decision,
        "threshold": round(threshold, 4),
        "signals": signals,
    }
