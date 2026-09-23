"""
LorentzNet + ArcFace Prototype Layer for Top Tagging
=====================================================
Lorentz-equivariant graph backbone (Gong et al., JHEP 07 (2022) 030,
arXiv:2201.08187) with the SAME ArcFace hyperspherical prototype head as
``efn_arcface.TopTaggingModel``, so the full downstream pipeline (spectral
clustering, eigengap, DGLAP fits, prototype analysis) is reusable unchanged.

Differences from the official implementation (kept deliberately):
  * Batched, padded (B, N, 4) inputs with a padding mask, instead of ragged
    variable-length graphs.  Message aggregation is done with masked sums
    rather than ``index_add_``/``scatter`` ops, which are unreliable on
    torch-directml (AMD GPU).  The math is identical to the official
    ``unsorted_segment_sum`` on a fully-connected graph without self-loops.
  * The final layer freezes the coordinate update (``last_layer=True``), so the
    particle 4-vectors x are never updated at the last block — the same
    equivariance-preserving choice as the official code.

Interface contract (drop-in for ``TopTaggingModel``):
    forward(constituents, labels=None, mask=None)
        -> (logits, similarities, jet_emb)
    constituents: (B, N, 4) columns (E, px, py, pz)
    mask:         (B, N) 1 = real constituent, 0 = padding
    logits:       ArcFace logits (B, 2), scale s applied (margin during train)
    similarities: raw cosine to prototypes (B, 2)
    jet_emb:      raw (unnormalized) embedding (B, embedding_dim)

NOTE (this machine): torch-directml only (AMD RX 5700 XT), torch 2.3.x,
numpy<2.  Keep everything float32; avoid index/scatter ops.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from arcefn.models.efn_arcface import ArcFaceHead, arcface_loss, count_parameters


def normsq4(p: torch.Tensor) -> torch.Tensor:
    """Minkowski square norm <p,p> = E^2 - px^2 - py^2 - pz^2 (component 0 = E)."""
    psq = p * p
    return 2 * psq[..., 0] - psq.sum(dim=-1)


def dotsq4(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Minkowski inner product <p,q> = E_p E_q - px_p px_q - py_p py_q - pz_p pz_q."""
    pq = p * q
    return 2 * pq[..., 0] - pq.sum(dim=-1)


def psi(p: torch.Tensor) -> torch.Tensor:
    """psi(p) = sign(p) * log(|p| + 1), the log-squashing used in LorentzNet."""
    return torch.sign(p) * torch.log(torch.abs(p) + 1)


class LGEB(nn.Module):
    """Lorentz Group Equivariant Block (official LorentzNet), batched/padded."""

    def __init__(self, n_hidden: int, n_scalar: int, c_weight: float = 1e-3, last_layer: bool = False):
        super().__init__()
        self.c_weight = c_weight
        self.last_layer = last_layer
        n_edge_attr = 2  # Minkowski norm + inner product

        self.phi_e = nn.Sequential(
            nn.Linear(n_hidden * 2 + n_edge_attr, n_hidden, bias=False),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
        )
        self.phi_m = nn.Sequential(
            nn.Linear(n_hidden, 1),
            nn.Sigmoid(),
        )
        self.phi_h = nn.Sequential(
            nn.Linear(n_hidden + n_hidden + n_scalar, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
        )
        if not last_layer:
            layer = nn.Linear(n_hidden, 1, bias=False)
            nn.init.xavier_uniform_(layer.weight, gain=0.001)
            self.phi_x = nn.Sequential(
                nn.Linear(n_hidden, n_hidden),
                nn.ReLU(),
                layer,
            )

    def forward(self, h: torch.Tensor, x: torch.Tensor, scalars: torch.Tensor,
                mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One message-passing block.

        Parameters
        ----------
        h : (B, N, H) node embeddings.
        x : (B, N, 4) particle 4-vectors (E, px, py, pz).
        scalars : (B, N, S) Lorentz-invariant node attributes.
        mask : (B, N) padding mask, 1 = real.

        Returns
        -------
        h : (B, N, H) updated embeddings.
        x : (B, N, 4) updated 4-vectors (unchanged at the last layer).
        """
        b, n, _ = x.shape

        hi = h.unsqueeze(2).expand(b, n, n, -1)   # (B,N,N,H) h_i
        hj = h.unsqueeze(1).expand(b, n, n, -1)   # (B,N,N,H) h_j
        xi = x.unsqueeze(2).expand(b, n, n, -1)   # (B,N,N,4)
        xj = x.unsqueeze(1).expand(b, n, n, -1)   # (B,N,N,4)
        xd = xi - xj                              # x_i - x_j

        norms = psi(normsq4(xd))                  # (B,N,N)
        dots = psi(dotsq4(xi, xj))                # (B,N,N)

        e = torch.cat([hi, hj, norms.unsqueeze(-1), dots.unsqueeze(-1)], dim=-1)
        e_flat = e.reshape(b * n * n, -1)         # contiguous after cat

        m = self.phi_e(e_flat)                    # (B*N*N, H)
        w = self.phi_m(m)                         # (B*N*N, 1)
        m = (m * w).reshape(b, n, n, -1)          # (B,N,N,H)

        # Mask: only real pairs, no self-loops (pure float — DML-safe, no bool ops)
        valid = (mask.unsqueeze(2) * mask.unsqueeze(1)).unsqueeze(-1)  # (B,N,N,1) 0/1
        eye = torch.eye(n).view(1, n, n, 1).to(x.device)               # (1,N,N,1) 1 on diagonal
        valid = valid * (1.0 - eye)
        m = m * valid

        agg = m.sum(dim=1)                        # (B,N,H) = sum_j m_ij

        if not self.last_layer:
            trans = self.phi_x(m.reshape(b * n * n, -1)).reshape(b, n, n, 1) * xd
            trans = trans * valid.float()
            cnt = valid.float().sum(dim=1)        # (B,N,1)
            agg_t = trans.sum(dim=1) / cnt.clamp(min=1.0)
            x = x + self.c_weight * agg_t

        hcat = torch.cat([h, agg, scalars], dim=-1)  # (B,N,2H+S)
        dh = self.phi_h(hcat.reshape(b * n, -1)).reshape(b, n, -1)
        h = h + dh

        return h, x


class LorentzNetArcFace(nn.Module):
    """LorentzNet backbone + ArcFace hyperspherical prototype head.

    Parameters
    ----------
    s : float
        ArcFace scale factor (default 16.0).
    m : float
        Angular margin in radians (default 0.5).
    embedding_dim : int
        Jet embedding dimension fed to the ArcFace head (default 64).
    n_hidden : int
        Latent width of the LGEB blocks (official: 72).
    n_layers : int
        Number of LGEB blocks (official: 6).
    c_weight : float
        Coordinate-update scale c (official: 1e-3).
    dropout : float
        Dropout on the readout MLP (official: 0.2).
    """

    def __init__(self, s: float = 16.0, m: float = 0.5, embedding_dim: int = 64,
                 n_hidden: int = 72, n_layers: int = 6, c_weight: float = 1e-3,
                 dropout: float = 0.2):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.n_scalar = 1  # per-particle invariant: psi(<p,p>)

        self.embedding = nn.Linear(self.n_scalar, n_hidden)
        self.lgebs = nn.ModuleList([
            LGEB(n_hidden, self.n_scalar, c_weight=c_weight,
                 last_layer=(i == n_layers - 1))
            for i in range(n_layers)
        ])
        self.readout = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_hidden, embedding_dim),
        )
        self.arcface_head = ArcFaceHead(input_dim=embedding_dim, n_classes=2, s=s, m=m)

    def forward(self, constituents: torch.Tensor, labels: torch.Tensor | None = None,
                mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Constituents (E, px, py, pz) -> LorentzNet embedding -> ArcFace logits.

        Parameters
        ----------
        constituents : (B, N, 4) columns (E, px, py, pz).
        labels : (B,) class indices (training only, for the margin).
        mask : (B, N) 1 = real constituent.

        Returns
        -------
        logits : (B, 2) ArcFace logits (scale s applied).
        similarities : (B, 2) raw cosine similarities to prototypes.
        jet_emb : (B, embedding_dim) raw (unnormalized) embedding.
        """
        b, n, _ = constituents.shape
        if mask is None:
            mask = (constituents[:, :, 0] > 0).float()

        x = constituents
        scalars = psi(normsq4(x)).unsqueeze(-1)          # (B,N,1)

        h = self.embedding(scalars)                       # (B,N,H)
        for block in self.lgebs:
            h, x = block(h, x, scalars, mask)

        # Mean-pool over real constituents -> jet embedding
        h_masked = h * mask.unsqueeze(-1)
        cnt = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        jet_h = h_masked.sum(dim=1) / cnt                 # (B,H)

        jet_emb = self.readout(jet_h)                     # (B, embedding_dim)
        logits, similarities = self.arcface_head(jet_emb, labels)
        return logits, similarities, jet_emb


def lorentznet_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """ArcFace cross-entropy loss (same as the EFN pipeline)."""
    return arcface_loss(logits, labels)


if __name__ == '__main__':
    torch.manual_seed(42)
    model = LorentzNetArcFace(s=16.0, m=0.0, embedding_dim=64)
    print(f"LorentzNet + ArcFace params: {count_parameters(model):,}")
    x = torch.randn(2, 32, 4)
    x[:, :, 0] = x[:, :, 0].abs() + 0.1                    # E > 0
    mask = (x[:, :, 0] > 0).float()
    y = torch.randint(0, 2, (2,))
    logits, sims, emb = model(x, y, mask)
    print(f"Logits: {logits.shape}, Sims: {sims.shape}, Emb: {emb.shape}")
    print("Success!")
