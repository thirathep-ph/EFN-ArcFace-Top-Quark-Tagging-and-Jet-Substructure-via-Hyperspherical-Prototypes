"""
Model Loading Helper
====================
Single source of truth for loading trained models from an experiment
directory, with automatic architecture and dimension inference.

Supports all three EFN heads:
    - ``TopTaggingModel``      (ArcFace head)          — key ``arcface_head.class_centers``
    - ``TopTaggingLinear``     (linear head)           — key ``classifier.weight``
    - ``TopTaggingCosLinear``  (cosine head, no margin)— key ``class_centers``

Hyperparameters are read from ``config.json`` when present; when it is
absent, the embedding dimension is inferred from the checkpoint shape.
This avoids the recurring hardcoded ``embedding_dim=64`` bug that breaks
every analysis script when pointed at a d=2 (or any other) model.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from arcefn.models.efn_arcface import TopTaggingModel
from arcefn.utils.paths import MODEL_DIR


def read_model_config(model_dir: str | Path | None = None) -> dict[str, Any]:
    """
    Resolve model hyperparameters from ``config.json`` (falling back to
    canonical defaults and checkpoint-shape inference).

    Parameters
    ----------
    model_dir : str or Path or None
        Experiment directory containing ``config.json`` and ``checkpoint.pt``.
        ``None`` uses the module-level ``MODEL_DIR`` (ARCEFN_MODEL_DIR-aware).

    Returns
    -------
    dict
        Keys: ``model_dir``, ``config_path``, ``checkpoint_path``,
        ``architecture`` (``'arcface' | 'linear' | 'coslinear'``),
        ``embedding_dim``, ``particle_dim``, ``scale``, ``margin``.
    """
    model_dir = Path(model_dir) if model_dir is not None else MODEL_DIR
    model_dir = model_dir.resolve()

    config_path = model_dir / 'config.json'
    checkpoint_path = model_dir / 'checkpoint.pt'

    config: dict[str, Any] = {}
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No checkpoint.pt in {model_dir}; cannot resolve model config."
        )

    # Checkpoints may store tensors on the DirectML private-use backend;
    # registering it first is required for torch.load to succeed.
    try:
        import torch_directml  # noqa: F401
    except ImportError:
        pass

    sd = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)

    if 'arcface_head.class_centers' in sd:
        architecture = 'arcface'
        shape_key = 'arcface_head.class_centers'
    elif 'classifier.weight' in sd:
        architecture = 'linear'
        shape_key = 'classifier.weight'
    elif 'class_centers' in sd:
        architecture = 'coslinear'
        shape_key = 'class_centers'
    else:
        raise ValueError(
            f"Cannot infer architecture from checkpoint {checkpoint_path}: "
            "none of arcface_head.class_centers / classifier.weight / "
            "class_centers keys found."
        )

    embedding_dim = config.get('embedding_dim')
    if embedding_dim is None:
        embedding_dim = sd[shape_key].shape[1]

    return {
        'model_dir': model_dir,
        'config_path': config_path,
        'checkpoint_path': checkpoint_path,
        'architecture': architecture,
        'embedding_dim': int(embedding_dim),
        'particle_dim': int(config.get('particle_dim', 128)),
        'scale': float(config.get('scale', 16.0) or 16.0),
        'margin': float(config.get('margin', 0.5) if config.get('margin', 0.5) is not None else 0.5),
    }


def load_model_from_dir(
    model_dir: str | Path | None = None,
    device: torch.device | str | None = None,
) -> torch.nn.Module:
    """
    Load a trained model from an experiment directory in eval mode.

    Parameters
    ----------
    model_dir : str or Path or None
        Experiment directory. ``None`` uses module-level ``MODEL_DIR``.
    device : torch.device or str or None
        Device to place the model on. ``None`` keeps it on CPU.

    Returns
    -------
    torch.nn.Module
        The loaded model (``.eval()``); for the ArcFace head the config
        margin is re-applied via ``set_margin``.
    """
    info = read_model_config(model_dir)
    device = device if device is not None else torch.device('cpu')

    model: torch.nn.Module
    if info['architecture'] == 'arcface':
        model = TopTaggingModel(
            s=info['scale'],
            m=info['margin'],
            embedding_dim=info['embedding_dim'],
            particle_dim=info['particle_dim'],
        )
    elif info['architecture'] == 'linear':
        from arcefn.models.efn_linear import TopTaggingLinear
        model = TopTaggingLinear(
            embedding_dim=info['embedding_dim'],
            particle_dim=info['particle_dim'],
        )
    else:  # coslinear
        from arcefn.models.efn_coslinear import TopTaggingCosLinear
        model = TopTaggingCosLinear(
            embedding_dim=info['embedding_dim'],
            particle_dim=info['particle_dim'],
            s=info['scale'],
        )

    sd = torch.load(str(info['checkpoint_path']), map_location='cpu', weights_only=False)
    model.load_state_dict(sd, strict=True)
    model = model.to(device)
    model.eval()

    if info['architecture'] == 'arcface':
        arcface_head = getattr(model, 'arcface_head')
        arcface_head.set_margin(info['margin'])

    return model
