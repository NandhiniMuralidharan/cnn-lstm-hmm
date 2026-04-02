"""
CNN Baseline Model
Author: Nandhini Muralidharan

Baseline architecture for ablation studies. Implements spatial feature extraction
followed by temporal global average pooling. This model serves to quantify
classification performance based solely on appearance, omitting temporal
sequence modeling.
"""

import torch
import torch.nn as nn
from src.models.cnn import CNNFeatureExtractor

class CNNBaseline(nn.Module):
    """
    CNN-only classifier with temporal averaging.

    Architecture:
        1. Frame-level feature extraction (CNN)
        2. Global average pooling across the temporal dimension
        3. Multi-layer perceptron (MLP) classification head
    """

    def __init__(self, n_classes: int = 3, feat_dim: int = 64, dropout: float = 0.3):
        super().__init__()

        self.cnn = CNNFeatureExtractor(feat_dim=feat_dim, dropout=dropout)

        # Classification head applied to temporally averaged features
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feat_dim // 2, n_classes)
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Forward pass processing video sequences.

        Args:
            frames: Input tensor of shape (B, T, 1, H, W)

        Returns:
            logits: Class scores of shape (B, n_classes)
        """
        # Feature extraction: (B, T, feat_dim)
        features = self.cnn.extract_sequence(frames)

        # Temporal pooling: (B, feat_dim)
        # Averaging across T eliminates sequential order information
        pooled = features.mean(dim=1)

        # Output logit generation
        return self.classifier(pooled)

    def count_parameters(self) -> int:
        """Returns the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
