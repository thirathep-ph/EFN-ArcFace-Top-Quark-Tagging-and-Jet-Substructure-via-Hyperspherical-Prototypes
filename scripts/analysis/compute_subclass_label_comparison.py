"""Compare true-label vs predicted-label subclass clustering.

The paper's headline subclass analysis (Table 5, subclass sizes, ARI) is
produced by clustering on TRUE class labels, while the paper text says
"predicted class".  This script runs the IDENTICAL spectral-clustering +
eigengap pipeline on BOTH partitions and quantifies whether the two give the
same physical subclasses, so the canonical partition can be chosen honestly.

Everything is read from existing artifacts (no retraining):
  * embeddings.npz  -> 404k precomputed embeddings + true labels
  * rf_cache_test.npz -> cached FastJet features (zg = features[:, 8])
  * checkpoint.pt   -> class centers (predictions = argmax cos(theta))

Run: python -m scripts.analysis.compute_subclass_label_comparison [--model_dir DIR]
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

from arcefn.utils.clustering import spectral_clustering_subclass
from arcefn.utils.paths import EXPERIMENTS
from scripts.analysis.compute_dglap_fit import SUBCLASS_NAMES, fit_zg

FEATURES = EXPERIMENTS / "rf_cache_test.npz"
MAX_K = 5


def normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-10)


def load_centers(checkpoint: Path) -> np.ndarray:
    import torch
    try:
        import torch_directml  # noqa: F401  (registers PrivateUse1 so the checkpoint loads)
    except ImportError:
        pass
    sd = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    return normalize(sd["arcface_head.class_centers"].detach().cpu().numpy())


def cluster_partition(emb_norm: np.ndarray, class_mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Spectral cluster the given events and map largest cluster -> Core (0)."""
    labels, k = spectral_clustering_subclass(emb_norm[class_mask], max_k=MAX_K)
    counts = Counter(labels)
    order = sorted(counts.keys(), key=lambda c: -counts[c])
    mapped = np.array([order.index(l) if l in order else 0 for l in labels])
    return mapped, int(k)


def subclass_summary(emb_norm: np.ndarray, zg: np.ndarray, mask: np.ndarray, cls: int) -> tuple[dict, np.ndarray]:
    sub, k = cluster_partition(emb_norm, mask)
    out = {"k": k, "n": int(mask.sum()), "subclasses": {}}
    for idx in range(2):
        name = SUBCLASS_NAMES[2 * cls + idx]
        m = sub == idx
        fit = fit_zg(zg[mask][m])
        out["subclasses"][name] = {
            "n": int(m.sum()),
            **({"alpha": fit["alpha"], "beta": fit["beta"], "r2": fit["r2"],
                "n_fit": fit["n"], "A": fit["A"]} if fit else {"alpha": None}),
        }
    return out, sub


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="True vs predicted-label subclass comparison")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Model directory (embeddings.npz + checkpoint.pt). "
                             "Default: canonical 64-d model dir.")
    args = parser.parse_args()

    model_dir = Path(args.model_dir) if args.model_dir else None
    base_dir = model_dir if model_dir else EXPERIMENTS / "robustness_s16_m05"
    out_dir = model_dir if model_dir else EXPERIMENTS
    embeddings_path = base_dir / "embeddings.npz"
    checkpoint_path = base_dir / "checkpoint.pt"
    out_path = out_dir / "subclass_label_comparison.json"

    t0 = time.time()
    emb_np = np.load(embeddings_path)
    embeddings, labels = emb_np["embeddings"], emb_np["labels"]
    zg_all = np.load(FEATURES)["features"][:, 8]
    assert len(embeddings) == len(labels) == len(zg_all) == 404000

    centers = load_centers(checkpoint_path)
    emb_norm = normalize(embeddings)
    preds = np.argmax(emb_norm @ centers.T, axis=1)

    acc = float((preds == labels).mean())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tp = int(((preds == 1) & (labels == 1)).sum())
    print(f"Accuracy {acc:.4f}  CM=[[{tn},{fp}],[{fn},{tp}]]")

    report: dict = {"n_events": len(embeddings), "accuracy": acc,
                    "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
                    "true_partition": {}, "pred_partition": {}}

    true_sub = np.full(len(labels), -1, dtype=int)
    pred_sub = np.full(len(labels), -1, dtype=int)
    for cls, cls_name in [(0, "QCD"), (1, "Top")]:
        t_mask = labels == cls
        p_mask = preds == cls
        report["true_partition"][cls_name], true_sub[t_mask] = \
            subclass_summary(emb_norm, zg_all, t_mask, cls)
        report["pred_partition"][cls_name], pred_sub[p_mask] = \
            subclass_summary(emb_norm, zg_all, p_mask, cls)

    true_global = 2 * labels + true_sub
    pred_global = 2 * preds + pred_sub
    ok = (true_sub >= 0) & (pred_sub >= 0)
    report["comparison"] = {
        "ari_global_4way": float(adjusted_rand_score(true_global[ok], pred_global[ok])),
        "ari_qcd_on_correct": float(adjusted_rand_score(
            true_sub[labels == 0][preds[labels == 0] == 0],
            pred_sub[labels == 0][preds[labels == 0] == 0])),
        "ari_top_on_correct": float(adjusted_rand_score(
            true_sub[labels == 1][preds[labels == 1] == 1],
            pred_sub[labels == 1][preds[labels == 1] == 1])),
        "overlap_matrix_4x4": _overlap(true_global, pred_global),
    }

    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {out_path}\n[COMPLETE] ({time.time()-t0:.0f}s)")


def _overlap(a: np.ndarray, b: np.ndarray) -> list[list[int]]:
    k = max(int(a.max()), int(b.max())) + 1
    mat = np.zeros((k, k), dtype=int)
    for i in range(k):
        for j in range(k):
            mat[i, j] = int(((a == i) & (b == j)).sum())
    return mat.tolist()


if __name__ == "__main__":
    main()
