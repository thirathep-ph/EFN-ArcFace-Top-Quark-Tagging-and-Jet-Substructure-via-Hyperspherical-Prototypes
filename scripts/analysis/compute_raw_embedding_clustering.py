"""
Cluster the *full* embedding set with no label conditioning.

Unlike the per-class subclass analyses (true-label or predicted-label
partitions), this script runs spectral clustering on all test events at
once, so the discovered clusters reflect only the geometry of the
learned embedding space. The partition is then compared against the
true class labels and against the per-class 4-way subclass partitions.

Run from the repository root as a module::

    python -m scripts.analysis.compute_raw_embedding_clustering [--model_dir DIR]

Reads artifacts only (no forward pass):
    * ``<model_dir>/embeddings.npz``  (keys: ``embeddings``, ``labels``)
    * ``experiments/rf_cache_test.npz``  (``features[:, 8]`` = zg)
    * ``<model_dir>/checkpoint.pt``  (class centers -> predicted labels)

Writes:
    * ``<out_dir>/raw_embedding_clustering.json``
"""
import json
import time
from pathlib import Path
from collections import Counter

import numpy as np
from sklearn.metrics import adjusted_rand_score

from arcefn.utils.clustering import spectral_clustering_subclass
from arcefn.utils.paths import EXPERIMENTS
from scripts.analysis.compute_dglap_fit import SUBCLASS_NAMES, fit_zg

FEATURES = EXPERIMENTS / "rf_cache_test.npz"
MAX_K = 10


def normalize(x):
    """L2-normalize rows; safe against zero norms."""
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10)


def load_centers(checkpoint):
    """Load the ArcFace class centers from a checkpoint (on CPU)."""
    import torch
    try:
        import torch_directml  # noqa: F401  (registers PrivateUse1 backend)
    except ImportError:
        pass
    sd = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    centers = sd["arcface_head.class_centers"].detach().cpu().numpy()
    return normalize(centers)


def cluster_full(emb_norm, max_k=MAX_K):
    """Spectral clustering over the full embedding set (no conditioning)."""
    labels, n_clusters = spectral_clustering_subclass(emb_norm, max_k=max_k)
    # Canonicalize labels by descending cluster size (largest -> 0).
    counts = Counter(labels)
    order = {lab: i for i, lab in enumerate(sorted(counts, key=lambda l: -counts[l]))}
    return np.array([order[l] for l in labels]), n_clusters


def cluster_partition(emb_norm, class_mask, max_k=5):
    """Per-class spectral partition (Core/Edge), largest cluster -> 0."""
    labels, n_clusters = spectral_clustering_subclass(emb_norm[class_mask], max_k=max_k)
    if n_clusters < 2:
        return np.full(class_mask.sum(), -1, dtype=int)
    counts = Counter(labels)
    order = {lab: i for i, lab in enumerate(sorted(counts, key=lambda l: -counts[l]))}
    return np.array([order[l] for l in labels])


def _overlap(a, b):
    """Joint-count matrix of two integer label arrays."""
    k = int(max(a.max(), b.max())) + 1
    mat = np.zeros((k, k), dtype=int)
    for x, y in zip(a.tolist(), b.tolist()):
        mat[x, y] += 1
    return mat.tolist()


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Unbiased spectral clustering of the full embedding set")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Model directory (embeddings.npz + checkpoint.pt). "
                             "Default: canonical MODEL_DIR.")
    args = parser.parse_args()

    model_dir = Path(args.model_dir) if args.model_dir else None
    base_dir = model_dir if model_dir else EXPERIMENTS / "robustness_s16_m05"
    out_dir = model_dir if model_dir else EXPERIMENTS

    t0 = time.time()
    print("=" * 60)
    print("Raw (unbiased) embedding clustering")
    print("=" * 60)

    with np.load(base_dir / "embeddings.npz") as npz:
        embeddings = npz["embeddings"]
        labels = npz["labels"]
    with np.load(FEATURES) as npz:
        zg = npz["features"][:, 8]

    n = len(labels)
    assert embeddings.shape[0] == n == zg.shape[0] == 404_000, \
        f"Unexpected sizes: emb={embeddings.shape[0]} labels={n} zg={zg.shape[0]}"
    print(f"  Embeddings: {embeddings.shape}, N = {n}")

    centers = load_centers(base_dir / "checkpoint.pt")
    emb_norm = normalize(embeddings)
    preds = np.argmax(emb_norm @ centers.T, axis=1)
    acc = float((preds == labels).mean())
    tn, fp, fn, tp = ((preds == 0) & (labels == 0)).sum(), \
                     ((preds == 1) & (labels == 0)).sum(), \
                     ((preds == 0) & (labels == 1)).sum(), \
                     ((preds == 1) & (labels == 1)).sum()

    # 1. Unbiased clustering of the full set.
    print("\nStep 1: spectral clustering of the full embedding set...")
    raw_sub, raw_k = cluster_full(emb_norm)
    print(f"  k = {raw_k}")

    # 2. Reference 4-way partitions (true- and predicted-label based).
    true_global = np.full(n, -1, dtype=int)
    pred_global = np.full(n, -1, dtype=int)
    true_sub_all = np.full(n, -1, dtype=int)
    pred_sub_all = np.full(n, -1, dtype=int)
    for cls, cname in [(0, "QCD"), (1, "Top")]:
        t_mask = labels == cls
        p_mask = preds == cls
        ts = cluster_partition(emb_norm, t_mask)
        ps = cluster_partition(emb_norm, p_mask)
        true_global[t_mask] = 2 * cls + ts
        pred_global[p_mask] = 2 * cls + ps
        true_sub_all[t_mask] = ts
        pred_sub_all[p_mask] = ps

    # 3. Comparisons (raw vs classes, raw vs 4-way partitions).
    ari_vs_classes = float(adjusted_rand_score(raw_sub, labels))
    ari_vs_true4 = float(adjusted_rand_score(raw_sub, true_global))
    ari_vs_pred4 = float(adjusted_rand_score(raw_sub, pred_global))

    print(f"  ARI(raw, class labels)      = {ari_vs_classes:.4f}")
    print(f"  ARI(raw, true 4-way)        = {ari_vs_true4:.4f}")
    print(f"  ARI(raw, pred  4-way)       = {ari_vs_pred4:.4f}")

    # 4. Per-cluster DGLAP fits on zg.
    clusters = {}
    for c in range(int(raw_sub.max()) + 1):
        m = raw_sub == c
        name = f"Cluster_{c}"
        fit = fit_zg(zg[m])
        clusters[name] = {
            "n": int(m.sum()),
            "n_qcd": int((labels[m] == 0).sum()),
            "n_top": int((labels[m] == 1).sum()),
            "alpha": fit["alpha"] if fit else None,
            "beta": fit["beta"] if fit else None,
            "r2": fit["r2"] if fit else None,
            "n_fit": fit["n"] if fit else None,
            "A": fit["A"] if fit else None,
        }
        if fit:
            print(f"  {name}: n={m.sum():6d}  "
                  f"QCD={clusters[name]['n_qcd']:6d} Top={clusters[name]['n_top']:6d}  "
                  f"alpha={fit['alpha']:.3f} beta={fit['beta']:.3f} r2={fit['r2']:.4f}")

    report = {
        "n_events": int(n),
        "accuracy": acc,
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "k_raw": int(raw_k),
        "ari_vs_class_labels": ari_vs_classes,
        "ari_vs_true_4way": ari_vs_true4,
        "ari_vs_pred_4way": ari_vs_pred4,
        "overlap_raw_vs_true4": _overlap(raw_sub, true_global),
        "overlap_raw_vs_pred4": _overlap(raw_sub, pred_global),
        "clusters": clusters,
    }

    out_path = out_dir / "raw_embedding_clustering.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Saved: {out_path}")
    print(f"  Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()