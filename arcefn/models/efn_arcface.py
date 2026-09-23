"""
EFN (Energy Flow Network) + ArcFace Prototype Layer for Top Tagging
=====================================================================
Ante-hoc (Intrinsic) Interpretability Architecture:
    Constituents (E, px, py, pz) -> Centering/Rotation -> EFN -> jet_embedding
    jet_embedding -> ArcFace Prototype Layer -> Classification

EFN Logic:
    - MLP (Phi) sees ONLY angular coordinates (delta_eta, delta_phi).
    - Energy (E) is used as a weight for the sum pooling: Sum(E_i * Phi(eta_i, phi_i)).
    - This ensures IRC-safety and focuses interpretability on energy patterns.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# Note: For pure speed in forward passes, we use torch operations that mimic vector logic.

def get_efn_features(constituents, mask=None):
    """
    Transform raw Cartesian (E, px, py, pz) into EFN-ready features.
    Ensures centering and rotation for better interpretability.
    """
    E = constituents[:, :, 0]
    px = constituents[:, :, 1]
    py = constituents[:, :, 2]
    pz = constituents[:, :, 3]

    # 1. Basic Kinematics
    pt = torch.sqrt(px**2 + py**2 + 1e-10)
    eta = torch.asinh(pz / (pt + 1e-10))
    phi = torch.atan2(py, px)

    # 2. Jet-level 4-momentum for centering
    if mask is not None:
        m = mask.unsqueeze(-1).float()
        jet_p4 = (constituents * m).sum(dim=1)
    else:
        jet_p4 = constituents.sum(dim=1)

    # Use vector-like logic for centering
    j_px, j_py, j_pz = jet_p4[:, 1], jet_p4[:, 2], jet_p4[:, 3]
    j_pt = torch.sqrt(j_px**2 + j_py**2 + 1e-10)
    j_eta = torch.asinh(j_pz / (j_pt + 1e-10))
    j_phi = torch.atan2(j_py, j_px)

    # 3. Relative Coordinates (Angular Only for EFN)
    d_eta = eta - j_eta.unsqueeze(1)
    d_phi = phi - j_phi.unsqueeze(1)
    d_phi = (d_phi + np.pi) % (2 * np.pi) - np.pi # Wrap phi

    # EFN input is JUST geometry
    features = torch.stack([d_eta, d_phi], dim=2) # (B, N, 2)

    # Normalized energy weights for pooling
    z = E / (E.sum(dim=1, keepdim=True) + 1e-10)

    if mask is not None:
        features = features * mask.unsqueeze(-1).float()
        z = z * mask.float()

    return features, z


class EFN(nn.Module):
    """
    Energy Flow Network: IRC-safe set processing.
    """
    def __init__(self, input_dim=2, particle_dim=256, jet_dim=256):
        """Initialize EFN with Phi (per-particle) and Rho (jet-level) MLPs.

        Parameters
        ----------
        input_dim : int
            Number of input features per particle (default 2 for d_eta, d_phi).
        particle_dim : int
            Hidden dimension for Phi MLP (default 256).
        jet_dim : int
            Output embedding dimension (default 256, final layer of Rho compresses to 64).
        """
        super().__init__()
        # Phi: Per-particle MLP (sees only geometry)
        self.phi = nn.Sequential(
            nn.Linear(input_dim, particle_dim),
            nn.BatchNorm1d(particle_dim),
            nn.ReLU(),
            nn.Linear(particle_dim, particle_dim),
            nn.BatchNorm1d(particle_dim),
            nn.ReLU(),
            nn.Linear(particle_dim, particle_dim),
        )
        
        # Rho: Jet-level MLP — transform then compress
        self.rho = nn.Sequential(
            nn.Linear(particle_dim, particle_dim),  # 128 → 128: transform
            nn.BatchNorm1d(particle_dim),
            nn.ReLU(),
            nn.Linear(particle_dim, jet_dim),       # 128 → 64: compress to embedding
        )

    def forward(self, x, z, mask=None):
        """
        x: (B, N, 2) - [d_eta, d_phi]
        z: (B, N) - [energy weights]
        """
        b, n, d = x.shape
        x_flat = x.reshape(b * n, d)
        phi_out_flat = self.phi(x_flat)
        phi_out = phi_out_flat.reshape(b, n, -1) # (B, N, particle_dim)

        # IRC-Safe Energy Weighted Pooling
        # Sum(z_i * Phi(x_i))
        weighted_phi = phi_out * z.unsqueeze(-1)
        jet_emb = weighted_phi.sum(dim=1) # (B, particle_dim)

        # Jet-level processing
        jet_emb = self.rho(jet_emb)
        return jet_emb


class ArcFaceHead(nn.Module):
    """
    Standard ArcFace Head using Class Centers (Weights).
    Highly optimized, mathematically clean, and fits the standard ArcFace formulation.
    """
    def __init__(self, input_dim, n_classes=2, s=16.0, m=0.5):
        """Initialize ArcFace head with learnable class centers on the hypersphere.

        Parameters
        ----------
        input_dim : int
            Dimension of input embeddings (typically 64).
        n_classes : int
            Number of classes (default 2 for binary Top vs QCD).
        s : float
            Scale factor for softmax temperature (default 16.0).
        m : float
            Additive angular margin in radians (default 0.5).
        """
        super().__init__()
        self.n_classes = n_classes
        self.s = s
        self.m = m
        
        # Class weights (centers) on the hypersphere
        self.class_centers = nn.Parameter(torch.FloatTensor(n_classes, input_dim))
        nn.init.xavier_uniform_(self.class_centers)

    def set_margin(self, m):
        """Update the angular margin (used during margin warmup).

        Parameters
        ----------
        m : float
            New margin value in radians.
        """
        self.m = m

    def forward(self, x, labels=None):
        """Compute ArcFace logits with optional angular margin during training.

        Parameters
        ----------
        x : torch.Tensor
            Input embeddings of shape (B, input_dim), pre-L2-normalization.
        labels : torch.Tensor or None
            Ground-truth class indices of shape (B,). Required during training
            to apply the angular margin at the correct class index.

        Returns
        -------
        logits : torch.Tensor
            Scaled logits of shape (B, n_classes), with margin applied during training.
        cosine : torch.Tensor
            Raw cosine similarities of shape (B, n_classes), without scale or margin.
        """
        # Normalize inputs and class centers
        x_norm = F.normalize(x, p=2, dim=1)
        c_norm = F.normalize(self.class_centers, p=2, dim=1)
        
        # Cosine similarity between embeddings and class centers
        cosine = F.linear(x_norm, c_norm)  # Shape: (B, 2)
        
        if self.training and labels is not None:
            cosine_clamped = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
            theta = torch.acos(cosine_clamped)
            
            # Clamp theta + m to np.pi to prevent gradient reversal
            theta_m = torch.clamp(theta + self.m, max=np.pi)
            cosine_m = torch.cos(theta_m)
            
            # Build logits: for each sample, use cosine_m at the true class index
            # and cosine at all other indices.
            # DirectML does not support F.one_hot + scatter well, so we use
            # an explicit index-based construction instead.
            logits = cosine.clone()
            batch_indices = torch.arange(len(labels), device=labels.device)
            logits[batch_indices, labels] = cosine_m[batch_indices, labels]
        else:
            logits = cosine
            
        return logits * self.s, cosine


class TopTaggingModel(nn.Module):
    """EFN + ArcFace: ante-hoc interpretable jet tagging model.

    Combines an IRC-safe Energy Flow Network backbone with hyperspherical
    ArcFace prototypes for intrinsic interpretability.
    """

    def __init__(self, s=16.0, m=0.5, embedding_dim=64, particle_dim=128):
        """Initialize EFN backbone with ArcFace classification head.

        Parameters
        ----------
        s : float
            Scale factor for ArcFace logits (default 16.0).
        m : float
            Angular margin in radians (default 0.5).
        embedding_dim : int
            Dimension of the jet embedding (default 64).
        particle_dim : int
            Hidden dimension for the per-particle MLP (default 128).
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.efn = EFN(input_dim=2, particle_dim=particle_dim, jet_dim=embedding_dim)
        self.arcface_head = ArcFaceHead(
            input_dim=embedding_dim, n_classes=2, s=s, m=m
        )

    def forward(self, constituents, labels=None, mask=None):
        """Forward pass: constituents → EFN embedding → ArcFace logits.

        Parameters
        ----------
        constituents : torch.Tensor
            Batched constituents of shape (B, N, 4) with features
            (E, px, py, pz).
        labels : torch.Tensor or None
            Ground-truth class indices for ArcFace margin (training only).
        mask : torch.Tensor or None
            Padding mask of shape (B, N) where 1 = real constituent, 0 = padding.

        Returns
        -------
        logits : torch.Tensor
            ArcFace logits of shape (B, 2) with scale s applied.
        similarities : torch.Tensor
            Raw cosine similarities to class prototypes of shape (B, 2).
        jet_emb : torch.Tensor
            Raw (unnormalized) jet embedding of shape (B, embedding_dim).
        """
        coords, z = get_efn_features(constituents, mask)
        jet_emb = self.efn(coords, z, mask)
        logits, similarities = self.arcface_head(jet_emb, labels)
        return logits, similarities, jet_emb

def arcface_loss(logits, labels):
    """Compute ArcFace cross-entropy loss.

    Parameters
    ----------
    logits : torch.Tensor
        Scaled ArcFace logits of shape (B, n_classes) with margin applied.
    labels : torch.Tensor
        Ground-truth class indices of shape (B,).

    Returns
    -------
    torch.Tensor
        Scalar cross-entropy loss.
    """
    return F.cross_entropy(logits, labels)

def distance_corr_loss(x, y, mask=None):
    """
    NOTE: This function is defined but currently unused in training.
    Available for future use as a decorrelation regularizer.
    
    Computes the Distance Correlation (DisCo) between x and y.
    x: (B,) - model predictions/scores
    y: (B,) - physical parameter (e.g., jet mass)
    mask: (B,) - boolean mask to filter (e.g., only background/QCD events)
    """
    if mask is not None:
        x = x[mask]
        y = y[mask]
    
    n = x.size(0)
    if n <= 1:
        return torch.tensor(0.0, device=x.device, requires_grad=True)
    
    # Pairwise distance matrices
    x_dist = torch.abs(x.unsqueeze(0) - x.unsqueeze(1))
    y_dist = torch.abs(y.unsqueeze(0) - y.unsqueeze(1))
    
    # Double centering
    x_mean_row = x_dist.mean(dim=1, keepdim=True)
    x_mean_col = x_dist.mean(dim=0, keepdim=True)
    x_mean_all = x_dist.mean()
    A = x_dist - x_mean_row - x_mean_col + x_mean_all
    
    y_mean_row = y_dist.mean(dim=1, keepdim=True)
    y_mean_col = y_dist.mean(dim=0, keepdim=True)
    y_mean_all = y_dist.mean()
    B = y_dist - y_mean_row - y_mean_col + y_mean_all
    
    # Distance covariance
    dcov2 = (A * B).mean()
    
    # Distance variance
    dvarx2 = (A * A).mean()
    dvary2 = (B * B).mean()
    
    # Distance correlation
    dcor = torch.sqrt(dcov2 / (torch.sqrt(dvarx2 * dvary2) + 1e-10) + 1e-10)
    return dcor

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
    print("EFN + ArcFace Model (xAI, 1 Class Center Per Class)")
    model = TopTaggingModel(embedding_dim=64)
    x = torch.randn(4, 200, 4)
    y = torch.randint(0, 2, (4,))
    logits, sims, emb = model(x, y)
    print(f"Logits: {logits.shape}, Sims: {sims.shape}, Emb: {emb.shape}")
    # Compute sample mass from constituents for test
    E_tot = x[:, :, 0].sum(dim=1)
    px_tot = x[:, :, 1].sum(dim=1)
    py_tot = x[:, :, 2].sum(dim=1)
    pz_tot = x[:, :, 3].sum(dim=1)
    mass = torch.sqrt(torch.clamp(E_tot**2 - (px_tot**2 + py_tot**2 + pz_tot**2), min=1e-10))
    dcor = distance_corr_loss(logits[:, 1], mass)
    print(f"DisCo Loss (between top logits & mass): {dcor.item():.4f}")
    print("Success!")
