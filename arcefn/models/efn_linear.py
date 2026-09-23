"""
EFN + Linear + CrossEntropy (Standard Classifier Baseline)
==========================================================
Ablation A: Standard fully-connected classification head.

Architecture:
    EFN backbone (IRC-safe, same as main model)
    → Linear(embedding_dim, 2) + bias
    → CrossEntropyLoss

This is the standard classifier baseline. Unlike ArcFace:
    - Embedding is NOT normalized
    - Weights are NOT normalized  
    - Has bias term
    - No margin
    - No scale factor

Purpose: Isolate the effect of ArcFace's hyperspherical constraint
on classification performance.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from arcefn.models.efn_arcface import EFN, get_efn_features


class TopTaggingLinear(nn.Module):
    """
    EFN backbone + Standard Linear classifier (Wx + b + CE).
    """
    def __init__(self, embedding_dim=64, particle_dim=128):
        """Initialize EFN backbone with a linear classifier head.

        Parameters
        ----------
        embedding_dim : int
            Dimension of the jet embedding (default 64).
        particle_dim : int
            Hidden dimension for the per-particle MLP (default 128).
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.efn = EFN(input_dim=2, particle_dim=particle_dim, jet_dim=embedding_dim)
        self.classifier = nn.Linear(embedding_dim, 2)

    def forward(self, constituents, labels=None, mask=None):
        """Forward pass: constituents → EFN embedding → linear logits.

        Parameters
        ----------
        constituents : torch.Tensor
            Batched constituents of shape (B, N, 4).
        labels : torch.Tensor or None
            Ignored (for compatibility with ArcFace interface).
        mask : torch.Tensor or None
            Padding mask of shape (B, N).

        Returns
        -------
        logits : torch.Tensor
            Class logits of shape (B, 2).
        jet_emb : torch.Tensor
            Jet embedding of shape (B, embedding_dim).
        """
        coords, z = get_efn_features(constituents, mask)
        jet_emb = self.efn(coords, z, mask)          # (B, embedding_dim)
        logits = self.classifier(jet_emb)             # (B, 2) — Wx + b
        return logits, jet_emb


def count_parameters(model):
    """Count trainable parameters in a model.

    Parameters
    ----------
    model : nn.Module
        PyTorch model.

    Returns
    -------
    int
        Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    print("EFN + Linear + CE (Standard Classifier Baseline)")
    model = TopTaggingLinear(embedding_dim=64)
    x = torch.randn(4, 200, 4)
    y = torch.randint(0, 2, (4,))
    logits, emb = model(x)
    loss = F.cross_entropy(logits, y)
    print(f"Logits: {logits.shape}, Emb: {emb.shape}, Loss: {loss.item():.4f}")
    print(f"Params: {count_parameters(model):,}")
    print("Success!")
