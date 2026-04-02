"""
Hidden Markov Model Layer
Author: Nandhini Muralidharan

Implements a differentiable HMM layer in PyTorch for temporal modeling.
The layer sits atop a BiLSTM and provides essential HMM algorithms (Forward,
Backward, Viterbi) in log-space for numerical stability.

Key Features:
    - Learnable transition matrix (normalized via log_softmax)
    - Learnable initial state probabilities
    - Integrated emission network mapping LSTM states to state log-probabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class HMMLayer(nn.Module):
    """
    Differentiable HMM component for modeling class-specific temporal structures.
    """

    def __init__(
        self,
        input_size: int,
        n_states: int = 4,
        init_self_loop: float = 0.7
    ):
        super().__init__()
        self.input_size = input_size
        self.n_states = n_states

        # --- Emission Network ---
        # Transforms BiLSTM outputs into log-probabilities for each HMM state
        self.emission_net = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.Tanh(),
            nn.Linear(input_size // 2, n_states)
        )

        # --- Learnable Parameters ---
        # log_A: Transition matrix initialized with a left-to-right (Bakis) structure
        log_A_init = self._init_left_to_right(n_states, init_self_loop)
        self.log_A_raw = nn.Parameter(log_A_init)

        # log_pi: Initial state distribution heavily biased toward the first state
        log_pi_init = torch.full((n_states,), -1e9)
        log_pi_init[0] = 0.0
        self.log_pi_raw = nn.Parameter(log_pi_init)

    @staticmethod
    def _init_left_to_right(K: int, self_loop: float) -> torch.Tensor:
        """Initializes transition matrix restricted to stay or move to the next state."""
        A = torch.full((K, K), -1e9)
        for i in range(K):
            if i < K - 1:
                A[i, i] = torch.log(torch.tensor(self_loop))
                A[i, i + 1] = torch.log(torch.tensor(1.0 - self_loop))
            else:
                A[i, i] = 0.0  # Absorbing state
        return A

    @property
    def log_A(self) -> torch.Tensor:
        """Normalized log transition matrix (rows sum to 1)."""
        return F.log_softmax(self.log_A_raw, dim=1)

    @property
    def log_pi(self) -> torch.Tensor:
        """Normalized log initial probabilities."""
        return F.log_softmax(self.log_pi_raw, dim=0)

    def emission_log_probs(self, lstm_out: torch.Tensor) -> torch.Tensor:
        """Computes log P(observation | state) for the sequence."""
        scores = self.emission_net(lstm_out)
        return F.log_softmax(scores, dim=-1)

    # --- Core HMM Algorithms ---

    def forward_algorithm(self, log_B: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes sequence log-likelihood log P(X|class) using the forward pass."""
        B, T, K = log_B.shape
        device = log_B.device
        log_alpha = torch.zeros(B, T, K, device=device)

        # Initialization
        log_alpha[:, 0, :] = self.log_pi.unsqueeze(0) + log_B[:, 0, :]

        # Recursion
        for t in range(1, T):
            log_alpha_prev = log_alpha[:, t-1, :].unsqueeze(2)  # (B, K, 1)
            log_A_exp = self.log_A.unsqueeze(0)                 # (1, K, K)

            # logsumexp over previous states s' for log alpha_{t-1}(s') + log A(s', s)
            trans = log_alpha_prev + log_A_exp
            log_alpha[:, t, :] = log_B[:, t, :] + torch.logsumexp(trans, dim=1)

        # Termination
        log_prob = torch.logsumexp(log_alpha[:, -1, :], dim=1)
        return log_alpha, log_prob

    def backward_algorithm(self, log_B: torch.Tensor) -> torch.Tensor:
        """Computes backward variables log beta for state posterior analysis."""
        B, T, K = log_B.shape
        device = log_B.device
        log_beta = torch.zeros(B, T, K, device=device)

        # Initialization
        log_beta[:, -1, :] = 0.0

        # Recursion (backward in time)
        for t in range(T - 2, -1, -1):
            log_beta_next = log_beta[:, t+1, :].unsqueeze(1)    # (B, 1, K)
            log_B_next = log_B[:, t+1, :].unsqueeze(1)          # (B, 1, K)
            log_A_exp = self.log_A.unsqueeze(0)                 # (1, K, K)

            trans = log_A_exp + log_B_next + log_beta_next
            log_beta[:, t, :] = torch.logsumexp(trans, dim=2)

        return log_beta

    def state_posteriors(self, log_alpha: torch.Tensor, log_beta: torch.Tensor) -> torch.Tensor:
        """Computes the probability of being in state s at time t: P(q_t = s | X)."""
        log_gamma = log_alpha + log_beta
        log_gamma = log_gamma - torch.logsumexp(log_gamma, dim=2, keepdim=True)
        return torch.exp(log_gamma)

    def viterbi_decode(self, log_B: torch.Tensor) -> torch.Tensor:
        """Identifies the most likely sequence of hidden states."""
        B, T, K = log_B.shape
        device = log_B.device
        log_delta = torch.zeros(B, T, K, device=device)
        psi = torch.zeros(B, T, K, dtype=torch.long, device=device)

        # Initialization
        log_delta[:, 0, :] = self.log_pi.unsqueeze(0) + log_B[:, 0, :]

        # Recursion
        for t in range(1, T):
            candidates = log_delta[:, t-1, :].unsqueeze(2) + self.log_A.unsqueeze(0)
            best_prev_vals, best_prev_idx = candidates.max(dim=1)

            log_delta[:, t, :] = log_B[:, t, :] + best_prev_vals
            psi[:, t, :] = best_prev_idx

        # Backtracking
        best_path = torch.zeros(B, T, dtype=torch.long, device=device)
        best_path[:, -1] = log_delta[:, -1, :].argmax(dim=1)

        for t in range(T - 2, -1, -1):
            best_path[:, t] = psi[torch.arange(B, device=device), t+1, best_path[:, t+1]]

        return best_path

    def forward(self, lstm_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Class-specific HMM pass: returns log-likelihood, forward variables, and emission scores."""
        log_B = self.emission_log_probs(lstm_out)
        log_alpha, log_prob = self.forward_algorithm(log_B)
        return log_prob, log_alpha, log_B
