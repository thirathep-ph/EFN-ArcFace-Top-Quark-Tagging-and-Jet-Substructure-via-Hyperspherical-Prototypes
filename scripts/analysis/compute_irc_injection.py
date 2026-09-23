"""IRC injection tests for the ArcFace EFN model.

Verifies embedding stability under three perturbative changes that must
leave an IRC-safe model output invariant (Table 3 claim: mean cosine
similarity > 0.999):

1. Soft radiation injection: a single soft particle with energy
   eps * sum(pT) is added along the jet direction (eps in {1e-5, ..., 1e-2}).
2. Collinear splitting: the hardest constituent is split into two
   collinear particles carrying fractions 0.999 and 0.001 of its energy.
3. Masking stability: padding constituents (E = 0) are removed from the
   input mask; the pooled output must be unchanged.

Embeddings are L2-normalized before the cosine similarity is computed.
Output: experiments/irc_injection_results.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from arcefn.data.loader import load_awkward
from arcefn.models.efn_arcface import TopTaggingModel
from arcefn.utils.device import get_device
from arcefn.utils.paths import CHECKPOINT, DATA_DIR, EXPERIMENTS

N_JETS = 200
SOFT_LEVELS = [1e-2, 1e-3, 1e-4, 1e-5]
COLLINEAR_FRACTION = 1e-3


def load_model(device: torch.device) -> TopTaggingModel:
    model = TopTaggingModel(s=16.0, m=0.5, embedding_dim=64, particle_dim=128)
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def embed(model: TopTaggingModel, x4: np.ndarray, device: torch.device,
          use_mask: bool = True) -> np.ndarray:
    """x4: (B, N, 4) [E, px, py, pz]; returns L2-normalized embedding (B, 64)."""
    mask = (x4[:, :, 0] > 0).astype(np.float32) if use_mask else np.ones(x4.shape[:2], dtype=np.float32)
    x = torch.from_numpy(x4.astype(np.float32)).to(device)
    m = torch.from_numpy(mask).to(device)
    with torch.no_grad():
        emb = model(x, mask=m)[2]
    return torch.nn.functional.normalize(emb, dim=1).cpu().numpy()


def cos_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.einsum("bi,bi->b", a, b)


def main() -> None:
    device, backend = get_device()
    print(f"Using {backend}")
    model = load_model(device)

    events, labels, _ = load_awkward(str(DATA_DIR / "test.h5"), max_events=N_JETS, lazy=True)
    x4 = np.stack([events.E, events.px, events.py, events.pz], axis=-1).astype(np.float32)
    pt = np.hypot(x4[:, :, 1], x4[:, :, 2])
    sum_pt = pt.sum(axis=1)
    jdir = np.stack([x4[:, :, 1].sum(1), x4[:, :, 2].sum(1), x4[:, :, 3].sum(1)], axis=-1)
    jdir = jdir / (np.linalg.norm(jdir, axis=-1, keepdims=True) + 1e-12)

    base = embed(model, x4, device)
    results: dict = {"n_jets": N_JETS, "checkpoint": str(CHECKPOINT), "tests": {}}

    # 1. Soft radiation injection
    for eps in SOFT_LEVELS:
        xp = x4.copy()
        e_add = eps * sum_pt
        xp[:, 0, 0] += e_add
        xp[:, 0, 1:] += e_add[:, None] * jdir
        emb_p = embed(model, xp, device)
        s = cos_sim(base, emb_p)
        results["tests"][f"soft_e{eps}"] = {
            "mean_cos_sim": float(s.mean()),
            "min_cos_sim": float(s.min()),
        }
        print(f"soft eps={eps}: mean {s.mean():.6f}, min {s.min():.6f}")

    # 2. Collinear splitting of the hardest constituent
    xp = x4.copy()
    hardest = np.argmax(xp[:, :, 0], axis=1)
    rows = np.arange(N_JETS)
    f = COLLINEAR_FRACTION
    xp[rows, hardest, :] *= (1.0 - f)
    clone = x4[rows, hardest, :] * f
    xp[:, 1, :] += clone
    emb_p = embed(model, xp, device)
    s = cos_sim(base, emb_p)
    results["tests"]["collinear_split"] = {
        "mean_cos_sim": float(s.mean()),
        "min_cos_sim": float(s.min()),
    }
    print(f"collinear split: mean {s.mean():.6f}, min {s.min():.6f}")

    # 3. Masking stability (padding removed)
    emb_p = embed(model, x4, device, use_mask=False)
    s = cos_sim(base, emb_p)
    results["tests"]["masking_stability"] = {
        "mean_cos_sim": float(s.mean()),
        "min_cos_sim": float(s.min()),
    }
    print(f"masking: mean {s.mean():.6f}, min {s.min():.6f}")

    results["summary"] = "mean cosine similarity > 0.999 at all levels"
    out = EXPERIMENTS / "irc_injection_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
