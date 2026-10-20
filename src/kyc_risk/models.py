"""Neural network architectures for multimodal KYC risk prediction and ablations."""

from __future__ import annotations

import torch
from torch import nn


class ImageEncoder(nn.Module):
    """Small CNN encoder for synthetic document image feature extraction."""

    def __init__(self) -> None:
        super().__init__()
        self.conv_blocks = nn.Sequential(
            # Block 1: 1 -> 16 channels (96x96 -> 48x48)
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2: 16 -> 32 channels (48x48 -> 24x24)
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 3: 32 -> 64 channels (24x24 -> 12x12)
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Global average pooling -> 64-d
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 64-d feature representation from input image tensor (B, 1, 96, 96)."""
        return self.conv_blocks(x)


class TabularEncoder(nn.Module):
    """MLP encoder transforming tabular risk factors into a 32-d latent representation."""

    def __init__(self, in_features: int = 9) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 32-d feature representation from tabular tensor (B, in_features)."""
        return self.mlp(x)


class FusionModel(nn.Module):
    """Multimodal fusion network combining document image CNN features and tabular MLP features."""

    def __init__(self, tabular_dim: int = 9) -> None:
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.tabular_encoder = TabularEncoder(in_features=tabular_dim)
        self.head = nn.Sequential(
            nn.Linear(64 + 32, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x_img: torch.Tensor, x_tab: torch.Tensor) -> torch.Tensor:
        """Compute risk logits given image and tabular modalities."""
        img_feats = self.image_encoder(x_img)
        tab_feats = self.tabular_encoder(x_tab)
        fused = torch.cat([img_feats, tab_feats], dim=1)
        return self.head(fused).squeeze(-1)


class ImageOnlyModel(nn.Module):
    """Ablation baseline: Document image only prediction model."""

    def __init__(self) -> None:
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x_img: torch.Tensor, x_tab: torch.Tensor | None = None) -> torch.Tensor:
        """Compute risk logits using only the document image."""
        feats = self.image_encoder(x_img)
        return self.head(feats).squeeze(-1)


class TabularOnlyModel(nn.Module):
    """Ablation baseline: Tabular risk features only prediction model."""

    def __init__(self, tabular_dim: int = 9) -> None:
        super().__init__()
        self.tabular_encoder = TabularEncoder(in_features=tabular_dim)
        self.head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(16, 1),
        )

    def forward(
        self,
        x_tab_or_img: torch.Tensor,
        x_tab: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute risk logits using only tabular attributes."""
        tensor = x_tab if x_tab is not None else x_tab_or_img
        feats = self.tabular_encoder(tensor)
        return self.head(feats).squeeze(-1)
