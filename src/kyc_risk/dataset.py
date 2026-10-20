"""PyTorch Dataset and tabular standardisation utilities for multimodal KYC."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

TABULAR_COLUMNS: list[str] = [
    "age",
    "account_age_days",
    "country_risk",
    "device_risk",
    "num_failed_logins_30d",
    "selfie_match_score",
    "address_mismatch",
    "pep_flag",
    "velocity_signups_same_device",
]


def fit_scaler(train_df: pd.DataFrame) -> dict[str, Any]:
    """Compute per-feature mean and standard deviation from training tabular data.

    Args:
        train_df: Training DataFrame containing tabular applicant features.

    Returns:
        Dictionary storing column order, mean, and std for each feature.
    """
    scaler: dict[str, Any] = {
        "columns": [col for col in TABULAR_COLUMNS if col in train_df.columns],
        "mean": {},
        "std": {},
    }
    for col in scaler["columns"]:
        mean_val = float(train_df[col].mean())
        std_val = float(train_df[col].std())
        scaler["mean"][col] = mean_val
        scaler["std"][col] = std_val if std_val > 1e-6 else 1.0
    return scaler


def transform_tabular(
    tabular_input: pd.DataFrame | dict[str, Any] | list[dict[str, Any]],
    scaler: dict[str, Any],
) -> np.ndarray:
    """Standardise tabular inputs using previously fitted scaler parameters.

    Args:
        tabular_input: DataFrame, single dict, or list of dicts.
        scaler: Fitted scaler dictionary with column means and standard deviations.

    Returns:
        Standardised float32 NumPy array of shape (N, num_features).
    """
    if isinstance(tabular_input, dict):
        df = pd.DataFrame([tabular_input])
    elif isinstance(tabular_input, list):
        df = pd.DataFrame(tabular_input)
    else:
        df = tabular_input.copy()

    columns = scaler.get("columns", TABULAR_COLUMNS)
    matrix = np.zeros((len(df), len(columns)), dtype=np.float32)

    for idx, col in enumerate(columns):
        if col in df.columns:
            vals = df[col].to_numpy(dtype=np.float32)
        else:
            vals = np.zeros(len(df), dtype=np.float32)
        mean_val = scaler["mean"].get(col, 0.0)
        std_val = scaler["std"].get(col, 1.0)
        matrix[:, idx] = (vals - mean_val) / std_val

    return matrix


class KYCDataset(Dataset):
    """PyTorch Dataset for multimodal KYC data returning image, tabular, and label tensors."""

    def __init__(
        self,
        images: np.ndarray,
        tabular: pd.DataFrame | np.ndarray,
        labels: np.ndarray | None = None,
        scaler: dict[str, Any] | None = None,
    ) -> None:
        """Initialize dataset.

        Args:
            images: Float32 array of shape (N, 1, 96, 96).
            tabular: Raw DataFrame (standardised via scaler) or pre-standardised array (N, 9).
            labels: Optional binary labels of shape (N,).
            scaler: Optional scaler dict if tabular is a raw DataFrame.
        """
        self.images = images.astype(np.float32)

        if isinstance(tabular, pd.DataFrame):
            if scaler is None:
                raise ValueError("Scaler required when passing raw tabular DataFrame.")
            self.tabular = transform_tabular(tabular, scaler)
        else:
            self.tabular = tabular.astype(np.float32)

        self.labels = labels.astype(np.float32) if labels is not None else None

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        img_tensor = torch.from_numpy(self.images[idx])
        tab_tensor = torch.from_numpy(self.tabular[idx])

        if self.labels is not None:
            label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)
            return img_tensor, tab_tensor, label_tensor

        return img_tensor, tab_tensor
