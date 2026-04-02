"""
BiLSTM Temporal Encoder
Author: Nandhini Muralidharan

Implements a Bidirectional LSTM to capture temporal dependencies within
feature sequences. Processes per-frame CNN embeddings to produce
context-aware representations for classification and HMM emission modeling.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

class BiLSTMEncoder(nn.Module):
    """
    Bidirectional LSTM encoder with support for sequence packing and
    optional feature projection.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        proj_size: int = 0
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = 2 * hidden_size

        # Core recurrent architecture
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.output_dropout = nn.Dropout(dropout)

        # Optional linear projection for HMM compatibility
        if proj_size > 0:
            self.projection = nn.Sequential(
                nn.Linear(2 * hidden_size, proj_size),
                nn.LayerNorm(proj_size),
                nn.Tanh()
            )
            self.output_size = proj_size
        else:
            self.projection = None

    def forward(
        self,
        features: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: (B, T, input_size)
            lengths: (B,) Optional sequence lengths for variable T.

        Returns:
            all_hidden: (B, T, output_size) Full sequence states.
            last_hidden: (B, output_size) Final context summary.
        """
        B, T, _ = features.shape

        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                features, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, (h_n, _) = self.lstm(packed)
            all_hidden, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True, total_length=T
            )
        else:
            all_hidden, (h_n, _) = self.lstm(features)

        # Extract the final hidden state from the top layer for both directions
        last_fwd = h_n[-2]
        last_bwd = h_n[-1]
        last_hidden = torch.cat([last_fwd, last_bwd], dim=-1)

        all_hidden = self.output_dropout(all_hidden)
        last_hidden = self.output_dropout(last_hidden)

        if self.projection is not None:
            all_hidden = self.projection(all_hidden)
            last_hidden = self.projection(last_hidden.unsqueeze(1)).squeeze(1)

        return all_hidden, last_hidden

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class CNNLSTMBaseline(nn.Module):
    """
    CNN-BiLSTM sequential classifier for ablation studies.
    Captures temporal ordering and long-range dependencies without
    explicit probabilistic state modeling.
    """

    def __init__(
        self,
        n_classes: int = 3,
        feat_dim: int = 64,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        from src.models.cnn import CNNFeatureExtractor

        self.cnn = CNNFeatureExtractor(feat_dim=feat_dim, dropout=dropout)
        self.lstm = BiLSTMEncoder(
            input_size=feat_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )

        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_classes)
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # Spatial encoding: (B, T, feat_dim)
        features = self.cnn.extract_sequence(frames)

        # Temporal encoding: (B, 2*hidden_size)
        _, last_hidden = self.lstm(features)

        # Sequential classification
        return self.classifier(last_hidden)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
