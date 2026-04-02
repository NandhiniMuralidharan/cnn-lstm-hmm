"""
PyTorch Dataset and DataLoader Utilities
Author: Nandhini Muralidharan

Handles sequence loading, normalization, and augmentation for synthetic
gesture data. Implements weak supervision by excluding hidden state paths
from the training tensors.
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Dict

class GestureDataset(Dataset):
    """
    Encapsulates synthetic gesture sequences for PyTorch-based modeling.
    """

    def __init__(
        self,
        data_dict: Dict,
        mean: float = 0.0,
        std: float = 1.0,
        augment: bool = False
    ):
        self.frames = data_dict['frames']
        self.labels = data_dict['labels']
        self.state_paths = data_dict['state_paths']
        self.mean = mean
        self.std = std
        self.augment = augment
        self.N, self.T, self.H, self.W = self.frames.shape

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a normalized frame sequence and its corresponding class label.
        """
        # Load sequence; copy to prevent modification of source data
        frames = self.frames[idx].copy()
        label = int(self.labels[idx])

        if self.augment:
            frames = self._augment(frames)

        # Standardize using training set statistics
        frames = (frames - self.mean) / (self.std + 1e-8)

        # Add channel dimension for CNN compatibility (T, 1, H, W)
        frames = frames[:, np.newaxis, :, :]

        return torch.from_numpy(frames), torch.tensor(label, dtype=torch.long)

    def get_state_path(self, idx: int) -> np.ndarray:
        """Retrieves ground-truth state paths for post-training analysis."""
        return self.state_paths[idx]

    def _augment(self, frames: np.ndarray) -> np.ndarray:
        """Applies temporally consistent spatial and intensity transformations."""
        # Horizontal flip
        if np.random.rand() < 0.5:
            frames = frames[:, :, ::-1].copy()

        # Intensity jitter
        shift = np.random.uniform(-0.05, 0.05)
        frames = np.clip(frames + shift, 0.0, 1.0)

        # Temporal jitter: duplicate neighbor to simulate minor timing variation
        if np.random.rand() < 0.3:
            t_drop = np.random.randint(1, self.T - 1)
            frames[t_drop] = frames[t_drop - 1]

        return frames

def compute_normalization_stats(train_data_dict: Dict) -> Tuple[float, float]:
    """Calculates dataset-wide mean and standard deviation for normalization."""
    frames = train_data_dict['frames']
    return float(frames.mean()), float(frames.std())

def build_dataloaders(
    dataset_path: str,
    batch_size: int = 16,
    num_workers: int = 2,
    augment_train: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Initializes DataLoaders for train, validation, and test splits.
    """
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)

    meta = dataset['meta']
    mean, std = compute_normalization_stats(dataset['train'])

    train_ds = GestureDataset(dataset['train'], mean, std, augment=augment_train)
    val_ds = GestureDataset(dataset['val'], mean, std, augment=False)
    test_ds = GestureDataset(dataset['test'], mean, std, augment=False)

    use_pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=use_pin, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=use_pin
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=use_pin
    )

    return train_loader, val_loader, test_loader, meta
