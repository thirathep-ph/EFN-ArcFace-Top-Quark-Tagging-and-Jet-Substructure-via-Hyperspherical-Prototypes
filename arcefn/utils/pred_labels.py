"""Predicted labels (what the model sees) for subclass analyses.

Primary partition throughout the paper is per-PREDICTED-class clustering.
Predictions use no angular margin (inference rule), matching
``compute_subclass_label_comparison.py``.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path


def normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-10)


def arcface_preds_from_centers(embeddings: np.ndarray, checkpoint: Path | str) -> np.ndarray:
    """Argmax-cosine predictions from a checkpoint's class centers (margin-free)."""
    import torch
    try:
        import torch_directml  # noqa: F401  (registers PrivateUse1 so the checkpoint loads)
    except ImportError:
        pass
    sd = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    key = "arcface_head.class_centers" if "arcface_head.class_centers" in sd else "class_centers"
    centers = normalize(sd[key].detach().cpu().numpy())
    return np.argmax(normalize(embeddings) @ centers.T, axis=1)


def check_cm(preds: np.ndarray, labels: np.ndarray, tag: str = "") -> None:
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tp = int(((preds == 1) & (labels == 1)).sum())
    acc = (tn + tp) / len(labels)
    print(f"[preds{(' ' + tag) if tag else ''}] acc={acc:.4f} CM=[[{tn},{fp}],[{fn},{tp}]]")
