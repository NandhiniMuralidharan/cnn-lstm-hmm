"""
Complete CNN-BiLSTM-HMM Model
Author: Nandhini Muralidharan

Integrates spatial feature extraction, temporal encoding, and probabilistic
state modeling into a single weakly supervised framework. The model
classifies sequences by evaluating the log-likelihood of each class-specific
HMM and selecting the maximum.
"""

import torch
import torch.nn as nn
from typing import Tuple, List
from src.models.cnn import CNNFeatureExtractor
from src.models.lstm import BiLSTMEncoder
from src.models.hmm import HMMLayer

class CNNLSTMHMMModel(nn.Module):
    """
    Weakly supervised gesture recognition architecture.

    The model consists of a shared CNN-BiLSTM backbone and a set of
    class-specific HMM layers. Supervision is applied at the sequence level
    via the forward algorithm log-likelihood.
    """

    def __init__(
        self,
        n_classes: int = 3,
        n_states: int = 4,
        feat_dim: int = 64,
        hidden_size: int = 128,
        num_lstm_layers: int = 2,
        dropout: float = 0.3,
        init_self_loop: float = 0.7
    ):
        super().__init__()

        self.n_classes = n_classes
        self.n_states = n_states

        # Shared spatial feature extractor
        self.cnn = CNNFeatureExtractor(feat_dim=feat_dim, dropout=dropout)

        # Shared temporal context encoder
        # proj_size ensures the BiLSTM output matches the HMM input requirements
        self.lstm = BiLSTMEncoder(
            input_size=feat_dim,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            dropout=dropout,
            proj_size=hidden_size
        )

        # Class-specific Hidden Markov Models
        self.hmms = nn.ModuleList([
            HMMLayer(
                input_size=hidden_size,
                n_states=n_states,
                init_self_loop=init_self_loop
            )
            for _ in range(n_classes)
        ])

    def forward(
        self,
        frames: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Computes per-class log-likelihoods for a batch of sequences.

        Args:
            frames: Input tensor of shape (B, T, 1, H, W)

        Returns:
            log_likelihoods: Tensor of shape (B, C) containing P(X | class)
            all_log_B: List of log-emission probabilities for visualization
        """
        # Feature extraction across the temporal dimension
        features = self.cnn.extract_sequence(frames)

        # Recurrent temporal encoding
        lstm_out, _ = self.lstm(features)

        # Evaluation of class-specific HMMs
        log_likelihoods = []
        all_log_B = []

        for c in range(self.n_classes):
            log_prob, _, log_B = self.hmms[c](lstm_out)
            log_likelihoods.append(log_prob)
            all_log_B.append(log_B)

        # Consolidate log-likelihoods into a single tensor
        log_likelihoods = torch.stack(log_likelihoods, dim=1)

        return log_likelihoods, all_log_B

    def predict(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Performs classification based on maximum log-likelihood.
        """
        log_likelihoods, _ = self.forward(frames)
        return log_likelihoods.argmax(dim=1)

    def decode_states(self, frames: torch.Tensor, class_idx: int) -> torch.Tensor:
        """
        Decodes the most likely hidden state sequence using the Viterbi algorithm.
        """
        features = self.cnn.extract_sequence(frames)
        lstm_out, _ = self.lstm(features)
        log_B = self.hmms[class_idx].emission_log_probs(lstm_out)
        return self.hmms[class_idx].viterbi_decode(log_B)

    def count_parameters(self) -> int:
        """Returns the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
