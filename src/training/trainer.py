"""
Unified Training and Evaluation Engine
Author: Nandhini Muralidharan

Provides a generalized training loop compatible with CNN-only, CNN-BiLSTM,
and hybrid CNN-BiLSTM-HMM architectures.
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import numpy as np

def compute_loss(model, frames, labels, model_type):
    """Calculates Cross-Entropy loss based on model architecture output."""
    if model_type == 'cnn_lstm_hmm':
        log_likelihoods, _ = model(frames)
        # Use negative log-likelihood as the loss for weak supervision
        loss = F.cross_entropy(log_likelihoods, labels)
        predictions = log_likelihoods.argmax(dim=1)
    else:
        logits = model(frames)
        loss = F.cross_entropy(logits, labels)
        predictions = logits.argmax(dim=1)
    return loss, predictions

def train_one_epoch(model, loader, optimizer, device, model_type, grad_clip=1.0):
    """Executes a single training epoch with gradient clipping."""
    model.train()
    total_loss, total_correct, n_samples = 0.0, 0, 0

    for frames, labels in loader:
        frames, labels = frames.to(device), labels.to(device)
        optimizer.zero_grad()

        loss, preds = compute_loss(model, frames, labels, model_type)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * frames.size(0)
        total_correct += (preds == labels).sum().item()
        n_samples += frames.size(0)

    return total_loss / n_samples, total_correct / n_samples

@torch.no_grad()
def evaluate(model, loader, device, model_type):
    """Calculates validation metrics."""
    model.eval()
    total_loss, total_correct, n_samples = 0.0, 0, 0

    for frames, labels in loader:
        frames, labels = frames.to(device), labels.to(device)
        loss, preds = compute_loss(model, frames, labels, model_type)

        total_loss += loss.item() * frames.size(0)
        total_correct += (preds == labels).sum().item()
        n_samples += frames.size(0)

    return total_loss / n_samples, total_correct / n_samples

def train_model(
    model, model_type, train_loader, val_loader,
    n_epochs=40, lr=1e-3, weight_decay=1e-4, patience=7,
    device='cuda', save_dir=None, model_name='model'
):
    """Orchestrates the training process including Early Stopping and LR Scheduling."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
    best_loss = float('inf')
    wait = 0

    print(f"\n--- Training Profile: {model_name} ({model_type}) ---")

    for epoch in range(1, n_epochs + 1):
        start_time = time.time()

        train_l, train_a = train_one_epoch(model, train_loader, optimizer, device, model_type)
        val_l, val_a = evaluate(model, val_loader, device, model_type)

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_l)

        history['train_loss'].append(train_l); history['train_acc'].append(train_a)
        history['val_loss'].append(val_l); history['val_acc'].append(val_a)
        history['lr'].append(current_lr)

        best_found = val_l < best_loss
        status = " (New Best)" if best_found else ""

        print(f"Epoch {epoch:02d} | Train Loss: {train_l:.4f} | Val Loss: {val_l:.4f} | "
              f"Val Acc: {val_a:.1%} | LR: {current_lr:.2e} | Time: {time.time()-start_time:.1f}s{status}")

        if best_found:
            best_loss = val_l
            wait = 0
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                torch.save({'state': model.state_dict(), 'history': history},
                           os.path.join(save_dir, f'{model_name}_best.pt'))
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping triggered after {patience} epochs.")
                break

    return history
