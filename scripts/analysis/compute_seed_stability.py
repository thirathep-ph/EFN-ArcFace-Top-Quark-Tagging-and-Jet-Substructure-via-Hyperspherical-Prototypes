"""Seed-stability analysis of ArcFace subclass discovery.

Trains nothing: for each ArcFace seed (42, 123, 7) it computes test-set
embeddings (or reuses ``embeddings.npz`` when present), discovers the
per-class spectral subclasses, and reports subclass fractions, DGLAP
``P(z_g)`` fit exponents, and the agreement of the discovered structure
against the seed-42 reference (ARI/NMI, fractions, alpha).

This answers: does the 4-subclass story (QCD/Top x Core/Edge) and its
physics interpretation survive a change of training seed?

Artifacts
---------
``experiments/seed_stability_analysis.json``
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from arcefn.data.loader import load_awkward, JetTaggingDataset
from arcefn.utils.clustering import spectral_clustering_subclass
from arcefn.utils.device import get_device
from arcefn.utils.model_loading import load_model_from_dir
from arcefn.utils.paths import DATA_DIR, EXPERIMENTS
from scripts.analysis.compute_dglap_fit import SUBCLASS_NAMES, fit_zg

MAX_EVENTS = 404000
BATCH_SIZE = 2048

CANONICAL_SEED_DIRS = {
    42: EXPERIMENTS / "robustness_s16_m05",
    123: EXPERIMENTS / "arcface_seed_123",
    7: EXPERIMENTS / "arcface_seed_7",
}


def seed_dirs_for(model_dir: Path | None) -> dict[int, Path]:
    """Resolve per-seed model directories.

    With ``model_dir`` (e.g. ``experiments/ablation_dim_2``) the per-seed dirs
    are ``<model_dir>`` (seed 42) and ``<model_dir>_seed_<s>`` (other seeds);
    otherwise the canonical 64-d ArcFace directories are used.
    """
    if model_dir is None:
        return dict(CANONICAL_SEED_DIRS)
    parent = model_dir.parent
    name = model_dir.name
    return {
        42: model_dir,
        123: parent / f"{name}_seed_123",
        7: parent / f"{name}_seed_7",
    }


def embeddings_for(seed: int, seed_dirs: dict[int, Path], device: torch.device) -> np.ndarray:
    """Return test embeddings for ``seed``, computing them if not cached."""
    cache = seed_dirs[seed] / "embeddings.npz"
    if cache.exists():
        return np.load(cache)["embeddings"]

    ckpt_dir = seed_dirs[seed]
    if not (ckpt_dir / "checkpoint.pt").exists():
        raise FileNotFoundError(f"no checkpoint for seed {seed}: {ckpt_dir}")

    model = load_model_from_dir(ckpt_dir, device)
    model.eval()

    events, labels, weights = load_awkward(
        str(DATA_DIR / "test.h5"), max_events=MAX_EVENTS, lazy=False
    )
    ds = JetTaggingDataset(events, labels, weights)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    all_emb = []
    with torch.no_grad():
        for x, y, w, m, _ in loader:
            x, m = x.to(device), m.to(device)
            _, _, emb = model(x, mask=m)
            all_emb.append(emb.cpu().numpy())
    return np.vstack(all_emb)


def subclass_labels(embeddings: np.ndarray, cls_labels: np.ndarray) -> np.ndarray:
    """Spectral 2-subclass split per class; largest cluster -> Core."""
    sub = np.full(len(cls_labels), -1, dtype=np.int8)
    for cls in (0, 1):
        mask = cls_labels == cls
        if mask.sum() <= 10:
            continue
        x = embeddings[mask]
        x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-10)
        sc_labels, _ = spectral_clustering_subclass(x, max_k=5)
        counts = Counter(sc_labels)
        order = sorted(counts.keys(), key=lambda c: -counts[c])
        mapped = np.array([order.index(l) if l in order else 0 for l in sc_labels])
        sub[mask] = mapped
    return sub


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7])
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Base model directory (e.g. experiments/ablation_dim_2). "
                             "Default: canonical 64-d ArcFace dirs.")
    args = parser.parse_args()

    t0 = time.time()
    device, _ = get_device()
    seed_dirs = seed_dirs_for(Path(args.model_dir) if args.model_dir else None)
    out_dir = Path(args.model_dir) if args.model_dir else EXPERIMENTS
    feat = np.load(EXPERIMENTS / "rf_cache_test.npz")
    cls_labels = feat["labels"].astype(int)
    zg_all = feat["features"][:, 8]
    assert len(cls_labels) == MAX_EVENTS

    seeds: dict[int, dict] = {}
    embed_dim = None
    for seed in args.seeds:
        print(f"=== Seed {seed} ({seed_dirs[seed].name}) ===")
        emb = embeddings_for(seed, seed_dirs, device)
        if embed_dim is None:
            embed_dim = emb.shape[1]
        assert emb.shape == (MAX_EVENTS, embed_dim), emb.shape
        sub = subclass_labels(emb, cls_labels)
        combined = (sub + 2 * cls_labels).astype(int)

        entry: dict = {"sub": sub.tolist(), "combined": combined.tolist()}
        frac = {}
        dglap = {}
        for sc in range(4):
            n = int(np.sum(combined == sc))
            frac[SUBCLASS_NAMES[sc]] = {"n": n, "pct": round(100.0 * n / MAX_EVENTS, 2)}
            fit = fit_zg(zg_all[combined == sc])
            if fit:
                dglap[SUBCLASS_NAMES[sc]] = {
                    "alpha": round(fit["alpha"], 4),
                    "beta": round(fit["beta"], 4),
                    "r2": round(fit["r2"], 4),
                    "n": fit["n"],
                }
            print(f"  {SUBCLASS_NAMES[sc]:9s} n={n:>7,} ({frac[SUBCLASS_NAMES[sc]]['pct']:5.2f}%)"
                  f"  alpha={dglap.get(SUBCLASS_NAMES[sc], {}).get('alpha', float('nan'))}")
        entry["fractions"] = frac
        entry["dglap"] = dglap
        del emb
        import gc
        gc.collect()
        seeds[seed] = entry

    # Agreement against the seed-42 reference
    ref_seed = 42
    agreement = {}
    if ref_seed in seeds:
        ref_combined = np.array(seeds[ref_seed]["combined"])
        for seed in sorted(seeds):
            if seed == ref_seed:
                continue
            comb = np.array(seeds[seed]["combined"])
            a: dict = {
                "global": {
                    "ari": round(float(adjusted_rand_score(ref_combined, comb)), 4),
                    "nmi": round(float(normalized_mutual_info_score(ref_combined, comb)), 4),
                }
            }
            for cls, name in ((0, "QCD"), (1, "Top")):
                m = cls_labels == cls
                a[name] = {
                    "ari": round(float(adjusted_rand_score(
                        np.array(seeds[ref_seed]["sub"])[m], np.array(seeds[seed]["sub"])[m])), 4),
                    "nmi": round(float(normalized_mutual_info_score(
                        np.array(seeds[ref_seed]["sub"])[m], np.array(seeds[seed]["sub"])[m])), 4),
                }
            print(f"[vs seed 42] seed {seed}: global ARI={a['global']['ari']} NMI={a['global']['nmi']} | "
                  f"QCD ARI={a['QCD']['ari']} | Top ARI={a['Top']['ari']}")
            agreement[seed] = a

    out = {
        "analysis": "seed-stability of ArcFace subclass discovery",
        "n_events": MAX_EVENTS,
        "per_seed": {
            str(k): {"fractions": v["fractions"], "dglap": v["dglap"]}
            for k, v in seeds.items()
        },
        "agreement_vs_seed42": {str(k): v for k, v in agreement.items()},
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    path = out_dir / "seed_stability_analysis.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {path}  ({out['elapsed_seconds']}s)")


if __name__ == "__main__":
    main()