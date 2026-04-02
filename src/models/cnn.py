"""
CNN Feature Extractor Module
Author: Nandhini Muralidharan

Converts individual frames into compact feature vectors. This module serves
as the spatial encoder for the CNN-LSTM-HMM temporal pipeline.
"""

import torch
import torch.nn as nn
from typing import Tuple

class ConvBlock(nn.Module):
    """
    Standard convolutional block comprising Convolution, Batch Normalization,
    ReLU activation, and Max Pooling.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pool_size: int = 2
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class CNNFeatureExtractor(nn.Module):
    """
    Frame-level encoder that extracts spatial features from grayscale sequences.
    Utilizes Global Average Pooling for spatial invariance and final linear
    projection for flexible feature sizing.
    """

    def __init__(self, feat_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.feat_dim = feat_dim

        # Encoder architecture: (1,32,32) -> (64,4,4)
        self.conv_blocks = nn.Sequential(
            ConvBlock(1, 16),
            ConvBlock(16, 32),
            ConvBlock(32, 64),
        )

        # Spatial aggregation and dimensionality reduction
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.projector = nn.Sequential(
            nn.Linear(64, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Processes a single batch of frames (B, 1, H, W)."""
        x = self.conv_blocks(x)
        x = self.gap(x).flatten(start_dim=1)
        return self.projector(x)

    def extract_sequence(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Processes full video sequences (B, T, 1, H, W) by flattening the temporal
        dimension for vectorized feature extraction.
        """
        B, T, C, H, W = frames.shape
        frames_flat = frames.view(B * T, C, H, W)

        feats_flat = self.forward(frames_flat)
        return feats_flat.view(B, T, self.feat_dim)

    def count_parameters(self) -> int:
        """Returns the number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
