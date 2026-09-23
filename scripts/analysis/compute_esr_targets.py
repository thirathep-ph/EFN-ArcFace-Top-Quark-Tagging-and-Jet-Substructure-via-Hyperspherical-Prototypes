"""POC: build symbolic-regression distillation targets for EFN+ArcFace.

Goal (user): replace the trained weights with closed-form expressions.
Pipeline distilled for PySR (Route B: raw constituents, no grooming):

  theta_i = (d_eta_i, d_phi_i)     per particle (deterministic, jet-axis-relative)
  z_i     = E_i / Sum E            energy weight
  phi_net = EFN.phi(theta_i)       (B,200,128)  the only genuine learned function
  jet     = Sum_i z_i * phi_net_i  pooled 128-d
  rho + cosine + scale             deterministic nonlinear readout (absorbed by G)

Distillation targets (exact-plane, decision-relevant):
  - P1 (per particle):  t_i = phi_net(theta_i) @ C_norm.T  (C_norm = normalized
                         ArcFace class centers, 2x64) -> symbolic phi_sym(theta)->t.
                         Since the decision depends only on the projection of the
                         pooled embedding onto the plane spanned by the two
                         prototypes, this projection loses no decision information
                         (unlike the earlier PCA-2D target, ~58% variance).
  - R1 (jet level):     (U,V) = Sum_i z_i t_i,  target = logit (model decision value)
                         -> symbolic readout G(U,V) -> logit  (absorbs rho+cos+s)

Artifacts (experiments/esr/):
  perparticle.npz : eta (M,), phi (M,), T (M,C)   [rows = real particles, seed-capped]
  jet.npz         : U (B,), V (B,), P (B,), logit (B,), labels (B,)
  meta.json       : n_events, seed, n_components, target description
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from arcefn.data.loader import load_awkward
from arcefn.models.efn_arcface import TopTaggingModel
from arcefn.utils.device import get_device
from arcefn.utils.paths import CHECKPOINT, DATA_DIR, EXPERIMENTS

N_EVENTS = 40000
BATCH = 512
SEED = 42
N_COMP = 2


def main() -> None:
    dev, _ = get_device()

    model = TopTaggingModel(s=16.0, m=0.5, embedding_dim=64, particle_dim=128)
    ck = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    model.load_state_dict(sd)
    model.to(dev)
    model.eval()

    events, labels, _ = load_awkward(str(DATA_DIR / "test.h5"), max_events=N_EVENTS, lazy=True)
    total = len(labels)
    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(total, min(N_EVENTS, total), replace=False))

    E = np.asarray(events.E, dtype=np.float32)
    px = np.asarray(events.px, dtype=np.float32)
    py = np.asarray(events.py, dtype=np.float32)
    pz = np.asarray(events.pz, dtype=np.float32)

    E = E[idx]
    px = px[idx]
    py = py[idx]
    pz = pz[idx]
    labels = labels[idx]
    n = len(labels)
    print(f"[esr] sampled {n} events (seed {SEED})")
    mask = (E > 0).astype(np.float32)

    X = np.stack([E, px, py, pz], axis=-1)
    pt = np.sqrt(px**2 + py**2 + 1e-10)
    eta = np.arcsinh(pz / (pt + 1e-10))
    phi = np.arctan2(py, px)
    jet_p4 = (X * mask[..., None]).sum(axis=1)
    j_pt = np.sqrt(jet_p4[:, 1] ** 2 + jet_p4[:, 2] ** 2 + 1e-10)
    j_eta = np.arcsinh(jet_p4[:, 3] / (j_pt + 1e-10))
    j_phi = np.arctan2(jet_p4[:, 2], jet_p4[:, 1])
    d_eta = eta - j_eta[:, None]
    d_phi = phi - j_phi[:, None]
    d_phi = (d_phi + np.pi) % (2 * np.pi) - np.pi
    coords = np.stack([d_eta, d_phi], axis=-1)
    z = E / (E.sum(axis=1, keepdims=True) + 1e-10)
    z = z * mask

    coords_t = torch.from_numpy(coords).to(dev)

    phi_net_all = []
    logits_all = []
    with torch.no_grad():
        for s in range(0, n, BATCH):
            e = min(s + BATCH, n)
            c = coords_t[s:e]
            fi = model.efn.phi(c.reshape(-1, 2)).reshape(e - s, -1, 128)
            phi_net_all.append(fi.cpu().numpy())
            zj = torch.from_numpy(z[s:e]).to(dev)
            out = model(torch.from_numpy(X[s:e]).to(dev), mask=torch.from_numpy(mask[s:e]).to(dev))
            logits_all.append(out[0].cpu().numpy())
    phi_net = np.concatenate(phi_net_all, axis=0)          # (N,200,128)
    logits = np.concatenate(logits_all, axis=0)            # (N,2)
    P = torch.softmax(torch.from_numpy(logits), dim=1).numpy()[:, 1]

    # ---- exact-plane projector: least-squares L: R^128 -> R^2 ----
    # cosines (decision-relevant) = logits / s  (s=16.0)
    cos = (logits / 16.0).astype(np.float64)               # (N,2)
    jet128 = (phi_net * z[:, :, None]).sum(axis=1)         # (N,128) pooled phi
    L, *_ = np.linalg.lstsq(jet128.astype(np.float64), cos, rcond=None)  # (128,2)
    cos_hat = jet128 @ L
    ss_res = float(np.sum((cos - cos_hat) ** 2))
    ss_tot = float(np.sum((cos - cos.mean(0)) ** 2))
    plane_r2 = 1.0 - ss_res / ss_tot
    comps = L.astype(np.float32)                           # (128,2)
    mean_ = np.zeros(128, dtype=np.float32)                # no centering: keep linearity
    print(f"[esr] exact-plane LS fit: cosine R^2 = {plane_r2:.4f}")

    out = EXPERIMENTS / "esr"
    out.mkdir(parents=True, exist_ok=True)

    # ---- per-particle rows (real particles only, seed-capped at 1M) ----
    real2 = z > 0                                          # (N,200)
    eta_rows, phi_rows, T_rows = [], [], []
    for i in range(n):
        idxs = np.where(real2[i])[0]
        if len(idxs) == 0:
            continue
        t_i = (phi_net[i, idxs] - mean_[None, :]).dot(comps)   # (k, C)
        eta_rows.append(d_eta[i, idxs])
        phi_rows.append(d_phi[i, idxs])
        T_rows.append(t_i)
    Eta = np.concatenate(eta_rows).astype(np.float32)
    Phi = np.concatenate(phi_rows).astype(np.float32)
    Tmat = np.concatenate(T_rows).astype(np.float32)
    rng3 = np.random.default_rng(SEED + 3)
    sel = rng3.choice(len(Eta), min(1_000_000, len(Eta)), replace=False)
    Eta, Phi, Tmat = Eta[sel], Phi[sel], Tmat[sel]

    # ---- jet coords (U,V) = Sum_i z_i t_i ----
    t3 = (phi_net - mean_[None, None, :]).dot(comps)     # (N,200,C)
    jetcoords = (t3 * z[:, :, None]).sum(axis=1)           # (N,C)

    np.savez_compressed(
        str(out / "perparticle.npz"),
        eta=Eta, phi=Phi, T=Tmat,
    )
    np.savez_compressed(
        str(out / "jet.npz"),
        U=jetcoords[:, 0], V=jetcoords[:, 1],
        P=P.astype(np.float32), logit=logits[:, 1].astype(np.float32),
        labels=labels.astype(np.int32),
    )
    meta = {
        "n_events": n,
        "seed": SEED,
        "n_components": N_COMP,
        "target": "exact-plane: least-squares L: pooled phi (128-d) -> (cos_QCD, cos_Top); per-particle t_i = L @ phi(x_i)",
        "plane_r2": plane_r2,
        "n_particles": int(len(Eta)),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[esr-targets] saved -> {out}")
    print(f"  exact-plane LS projector (C={N_COMP}), cosine R^2 = {plane_r2:.4f}")


if __name__ == "__main__":
    main()
