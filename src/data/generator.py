"""
Synthetic Video Sequence Generator
Author: Nandhini Muralidharan

Generates synthetic grayscale video sequences for weak supervision experiments.
Each sequence represents a class-specific temporal evolution modeled via
left-to-right Markov chains, simulating phases such as preparation, stroke,
and retraction.
"""

import numpy as np
import os
import pickle
from typing import Tuple, Dict, List

# --- Pattern Generation Functions ---

def _make_bright_pattern(H: int, W: int, intensity: float) -> np.ndarray:
    """Generates a uniform brightness pattern."""
    return np.full((H, W), intensity, dtype=np.float32)

def _make_horizontal_bar(H: int, W: int, position: float) -> np.ndarray:
    """Generates a horizontal bar with Gaussian spread at a specified vertical position."""
    frame = np.zeros((H, W), dtype=np.float32)
    bar_row = int(position * (H - 1))
    for r in range(H):
        frame[r, :] = np.exp(-0.5 * ((r - bar_row) / (H * 0.1)) ** 2)
    return frame

def _make_diagonal(H: int, W: int, angle: float) -> np.ndarray:
    """Generates a rotating diagonal stripe using a Gaussian line profile."""
    frame = np.zeros((H, W), dtype=np.float32)
    cx, cy = W / 2, H / 2
    for r in range(H):
        for c in range(W):
            dist = abs((c - cx) * np.sin(angle) - (r - cy) * np.cos(angle))
            frame[r, c] = np.exp(-0.5 * (dist / (W * 0.12)) ** 2)
    return frame

# --- Prototype Construction ---

def _build_class_prototypes(
    n_classes: int,
    n_states: int,
    H: int,
    W: int
) -> Dict[int, List[np.ndarray]]:
    """
    Constructs mean frame prototypes for each (class, state) pair.
    """
    prototypes = {}

    for c in range(n_classes):
        prototypes[c] = []

        if c == 0:
            # Class 0: Intensity pulse (dim -> bright -> dim)
            half = n_states // 2
            intensities = list(np.linspace(0.1, 0.9, half + 1)) + \
                          list(np.linspace(0.9, 0.1, n_states - half - 1))
            for s in range(n_states):
                proto = _make_bright_pattern(H, W, intensities[s])
                prototypes[c].append(proto)

        elif c == 1:
            # Class 1: Vertical sweep of a horizontal bar
            positions = np.linspace(0.1, 0.9, n_states)
            for s in range(n_states):
                proto = _make_horizontal_bar(H, W, positions[s])
                prototypes[c].append(proto)

        elif c == 2:
            # Class 2: Angular rotation of a diagonal stripe
            angles = np.linspace(0, np.pi / 2, n_states)
            for s in range(n_states):
                proto = _make_diagonal(H, W, angles[s])
                prototypes[c].append(proto)

        else:
            # Default: Random Gaussian blobs
            rng = np.random.RandomState(seed=c * 100)
            centers = [(rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8))
                       for _ in range(n_states)]
            for s in range(n_states):
                frame = np.zeros((H, W), dtype=np.float32)
                cy_pos, cx_pos = int(centers[s][0] * H), int(centers[s][1] * W)
                for r in range(H):
                    for col in range(W):
                        d2 = (r - cy_pos)**2 + (col - cx_pos)**2
                        frame[r, col] = np.exp(-d2 / (2 * (W * 0.15)**2))
                prototypes[c].append(frame)

    return prototypes

# --- Temporal Modeling ---

def _make_transition_matrix(n_states: int, self_loop_prob: float = 0.7) -> np.ndarray:
    """
    Constructs a left-to-right (Bakis) transition matrix.
    Ensures sequential state progression with an absorbing final state.
    """
    A = np.zeros((n_states, n_states), dtype=np.float32)
    for i in range(n_states):
        if i < n_states - 1:
            A[i, i] = self_loop_prob
            A[i, i + 1] = 1.0 - self_loop_prob
        else:
            A[i, i] = 1.0
    return A

# --- Sequence and Dataset Generation ---

def generate_sequence(
    class_idx: int,
    prototypes: Dict[int, List[np.ndarray]],
    transition_matrix: np.ndarray,
    seq_len: int,
    noise_std: float,
    rng: np.random.RandomState
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a single video sequence based on a sampled hidden state path.
    """
    n_states = transition_matrix.shape[0]
    H, W = prototypes[class_idx][0].shape

    state_path = np.zeros(seq_len, dtype=np.int32)
    current_state = 0

    for t in range(1, seq_len):
        current_state = rng.choice(n_states, p=transition_matrix[current_state])
        state_path[t] = current_state

    frames = np.zeros((seq_len, H, W), dtype=np.float32)
    for t in range(seq_len):
        mu = prototypes[class_idx][state_path[t]]
        noise = rng.normal(0, noise_std, size=(H, W)).astype(np.float32)
        frames[t] = np.clip(mu + noise, 0.0, 1.0)

    return frames, state_path

def generate_dataset(
    n_classes: int = 3,
    n_states: int = 4,
    n_train: int = 500,
    n_val: int = 100,
    n_test: int = 100,
    seq_len: int = 30,
    H: int = 32,
    W: int = 32,
    noise_std: float = 0.15,
    self_loop_prob: float = 0.7,
    seed: int = 42,
    save_dir: str = None
) -> Dict:
    """
    Generates a structured train/validation/test dataset of synthetic sequences.
    """
    rng = np.random.RandomState(seed)
    prototypes = _build_class_prototypes(n_classes, n_states, H, W)
    A = _make_transition_matrix(n_states, self_loop_prob)

    dataset = {}
    splits = [('train', n_train), ('val', n_val), ('test', n_test)]

    for split_name, n_per_class in splits:
        all_frames, all_labels, all_states = [], [], []

        for c in range(n_classes):
            for _ in range(n_per_class):
                frames, state_path = generate_sequence(
                    c, prototypes, A, seq_len, noise_std, rng
                )
                all_frames.append(frames)
                all_labels.append(c)
                all_states.append(state_path)

        all_frames = np.stack(all_frames)
        all_labels = np.array(all_labels)
        all_states = np.stack(all_states)

        idx = rng.permutation(len(all_labels))
        dataset[split_name] = {
            'frames': all_frames[idx],
            'labels': all_labels[idx],
            'state_paths': all_states[idx],
        }

    dataset['meta'] = {
        'n_classes': n_classes,
        'n_states': n_states,
        'seq_len': seq_len,
        'frame_shape': (H, W),
        'noise_std': noise_std,
        'transition_A': A,
        'prototypes': prototypes,
    }

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, 'synthetic_dataset.pkl')
        with open(path, 'wb') as f:
            pickle.dump(dataset, f)

    return dataset
