"""
EFN + Cosine Linear + CE (ArcFace without Margin)
==================================================
Ablation B: ArcFace with margin=0.

Architecture:
    EFN backbone (IRC-safe)
    → L2-normalize embedding
    → Cosine similarity to normalized class centers
    → Scale by s
    → CrossEntropyLoss

Same as ArcFace EXCEPT no angular margin (m=0).
This isolates the effect of the margin hyperparameter.

ArcFace:   logits = cos(θ + m) * s  (training) / cos(θ) * s (eval)
CosLinear: logits = cos(θ) * s      (always)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from arcefn.models.efn_arcface import EFN, get_efn_features


class TopTaggingCosLinear(nn.Module):
    """
    EFN backbone + Cosine similarity classifier (no margin).
    Embedding and class centers are L2-normalized (hypersphere).
    """
    def __init__(self, embedding_dim=64, particle_dim=128, s=16.0):
        """Initialize EFN backbone with cosine-similarity classifier (ArcFace without margin).

        Parameters
        ----------
        embedding_dim : int
            Dimension of the jet embedding (default 64).
        particle_dim : int
            Hidden dimension for the per-particle MLP (default 128).
        s : float
            Scale factor for the cosine logits (default 16.0).
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.s = s

        self.efn = EFN(input_dim=2, particle_dim=particle_dim, jet_dim=embedding_dim)

        # Class centers on the hypersphere (same as ArcFace)
        self.class_centers = nn.Parameter(torch.FloatTensor(2, embedding_dim))
        nn.init.xavier_uniform_(self.class_centers)

    def forward(self, constituents, labels=None, mask=None):
        """Forward pass: constituents → EFN → cosine similarity logits.

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
            Scaled cosine similarity logits of shape (B, 2).
        jet_emb : torch.Tensor
            Raw (unnormalized) jet embedding of shape (B, embedding_dim).
        """
        coords, z = get_efn_features(constituents, mask)
        jet_emb = self.efn(coords, z, mask)            # (B, embedding_dim)

        # Normalize both embeddings and class centers
        jet_norm = F.normalize(jet_emb, p=2, dim=1)
        c_norm = F.normalize(self.class_centers, p=2, dim=1)

        # Cosine similarity (no margin)
        cosine = F.linear(jet_norm, c_norm)             # (B, 2)
        logits = cosine * self.s

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
    print("EFN + Cosine Linear + CE (ArcFace without margin)")
    model = TopTaggingCosLinear(embedding_dim=64)
    x = torch.randn(4, 200, 4)
    y = torch.randint(0, 2, (4,))
    logits, emb = model(x)
    loss = F.cross_entropy(logits, y)
    print(f"Logits: {logits.shape}, Emb: {emb.shape}, Loss: {loss.item():.4f}")
    print(f"Params: {count_parameters(model):,}")
    print("Success!")
