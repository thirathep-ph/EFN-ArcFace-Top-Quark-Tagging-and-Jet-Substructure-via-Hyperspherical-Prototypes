"""Lund planes per discovered subclass, split by prediction status.

Primary partition is PREDICTED-label (what the model sees): predicted-QCD
subclasses contain TN+FN, predicted-Top subclasses TP+FP. Panels therefore
split QCD-Edge into TN vs FN and Top-Core into TP vs FP (under true labels
these pairings were TN vs FP and TP vs FN).

Complements the status-only Lund planes (lund_status_*.png) with the
subclass-resolved view that the paper's misclassification-localization claim
needs: every FP belongs to QCD-Edge and every FN to Top-Core, so the
physically interesting panels are

  * QCD-Core   (all, essentially all true negative)
  * QCD-Edge   true negative  vs  false positive
  * Top-Core   true positive  vs  false negative
  * Top-Edge   (all, essentially all true positive)

The subclass partition is the primary PREDICTED-label one (spectral
clustering per predicted class over 404k, what the model sees); ground-truth
partition is available via --partition true. Prediction status is argmax
cosine similarity against the cached ArcFace class centers (no forward pass).

Artifacts read:
  * MODEL_DIR/embeddings.npz  -> 404k embeddings + labels
  * MODEL_DIR/checkpoint.pt   -> class centers
  * data/top_tagging/test.h5                       -> physics sample

Output:
  * MODEL_DIR/lund_planes/subclass_*.png
  * paper/figures/lund_subclass_*.png
  * experiments/subclass_lund_planes.json  (per-panel counts)
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from shutil import copy2
import os

import matplotlib

matplotlib.use("Agg")
import numpy as np

from arcefn.data.loader import load_events_by_indices
from arcefn.utils.clustering import spectral_clustering_subclass
from arcefn.utils.physics import compute_features, get_lund_coordinates
from arcefn.utils.plotting import plot_lund_plane
from arcefn.utils.paths import EXPERIMENTS

TEST_H5 = EXPERIMENTS.parent / "data" / "top_tagging" / "test.h5"

MAX_K = 5
PHYS_PER_GROUP = 2000
PHYS_CAP = 15000

SUBCLASS_NAMES = {0: "QCD-Core", 1: "QCD-Edge", 2: "Top-Core", 3: "Top-Edge"}
STATUS = ["True Positive", "False Positive", "True Negative", "False Negative"]

# (subclass id, status filter or None, file stem, plot label)
# Predicted-label partition: QCD-Edge holds TN+FN, Top-Core holds TP+FP.
PANELS = [
    (0, None, "subclass_qcd_core", "QCD-Core (all)"),
    (1, "True Negative", "subclass_qcd_edge_tn", "QCD-Edge (true negative)"),
    (1, "False Negative", "subclass_qcd_edge_fn", "QCD-Edge (false negative)"),
    (2, "True Positive", "subclass_top_core_tp", "Top-Core (true positive)"),
    (2, "False Positive", "subclass_top_core_fp", "Top-Core (false positive)"),
    (3, None, "subclass_top_edge", "Top-Edge (all)"),
]


def normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-10)


def load_centers(checkpoint_path) -> np.ndarray:
    import torch
    try:
        import torch_directml  # noqa: F401  (registers PrivateUse1 so the checkpoint loads)
    except ImportError:
        pass

    sd = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return normalize(sd["arcface_head.class_centers"].detach().cpu().numpy())


def stratified_sample(indices: np.ndarray, n: int, rng: np.random.RandomState) -> np.ndarray:
    if len(indices) <= n:
        return indices
    return rng.choice(indices, size=n, replace=False)


def panel_stats(events) -> dict:
    ln_inv_delta, ln_kt = get_lund_coordinates(events)
    feats = compute_features(events)
    return {
        "ln_inv_delta_mean": float(np.mean(ln_inv_delta)),
        "ln_kt_mean": float(np.mean(ln_kt)),
        "mass_mean": float(np.mean(feats[:, 0])),
        "tau32_mean": float(np.mean(feats[:, 7])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate subclass-resolved Lund planes")
    parser.add_argument("--model_dir", type=str, default="experiments/robustness_s16_m05",
                        help="Directory containing model checkpoint and embeddings")
    parser.add_argument("--embedding_dim", type=int, default=None,
                        help="Embedding dimension (unused, kept for consistency)")
    parser.add_argument("--partition", type=str, default="predicted", choices=["predicted", "true"],
                        help="Cluster per predicted (model-seen, primary) or true labels")
    args = parser.parse_args()

    # Set paths based on model_dir
    model_dir = args.model_dir
    EMBEDDINGS = os.path.join(model_dir, "embeddings.npz")
    CHECKPOINT = os.path.join(model_dir, "checkpoint.pt")
    LUND_DIR = os.path.join(model_dir, "lund_planes")
    OUT_JSON = os.path.join(os.path.dirname(model_dir), "subclass_lund_planes.json")

    t0 = time.time()
    emb_np = np.load(EMBEDDINGS)
    embeddings, labels = emb_np["embeddings"], emb_np["labels"]
    assert len(embeddings) == 404_000, f"expected 404k, got {len(embeddings)}"

    centers = load_centers(CHECKPOINT)
    emb_norm = normalize(embeddings)
    preds = np.argmax(emb_norm @ centers.T, axis=1)

    partition_labels = preds if args.partition == "predicted" else labels
    true_global = np.full(len(labels), -1, dtype=int)
    for cls in (0, 1):
        mask = partition_labels == cls
        sub, _ = spectral_clustering_subclass(emb_norm[mask], max_k=MAX_K)
        counts = Counter(sub)
        order = sorted(counts, key=lambda c: -counts[c])
        mapped = np.array([order.index(s) for s in sub])
        true_global[mask] = 2 * cls + mapped

    tp = (preds == 1) & (labels == 1)
    fp = (preds == 1) & (labels == 0)
    tn = (preds == 0) & (labels == 0)
    fn = (preds == 0) & (labels == 1)
    status = np.empty(len(labels), dtype=object)
    status[tp], status[fp], status[tn], status[fn] = STATUS

    rng = np.random.RandomState(42)
    all_idx = np.arange(len(labels))
    parts = []
    for sc in range(4):
        sc_idx = all_idx[true_global == sc]
        if len(sc_idx) > 0:
            parts.append(stratified_sample(sc_idx, PHYS_PER_GROUP, rng))
    for cat in STATUS:
        st_idx = all_idx[status == cat]
        if len(st_idx) > 0:
            parts.append(stratified_sample(st_idx, PHYS_PER_GROUP, rng))
    phys_idx = np.unique(np.concatenate(parts))
    rng.shuffle(phys_idx)
    phys_idx = phys_idx[:PHYS_CAP]

    print(f"Loading {len(phys_idx)} physics events from test.h5 ...")
    phys_events = load_events_by_indices(TEST_H5, phys_idx)

    os.makedirs(os.path.join(model_dir, "lund_planes"), exist_ok=True)
    results: dict = {"n_events_total": int(len(labels)), "n_physics_sample": int(len(phys_idx)), "panels": {}}
    for sc, status_filter, stem, label in PANELS:
        cond = (true_global == sc) if status_filter is None else (
            (true_global == sc) & (status == status_filter)
        )
        n_total = int(cond.sum())
        panel_mask = cond[phys_idx]
        n_panel = int(panel_mask.sum())
        if n_panel == 0:
            results["panels"][stem] = {"subclass": SUBCLASS_NAMES[sc], "status": status_filter,
                                       "n_total": n_total, "n_plotted": 0}
            continue
        lund_path = os.path.join(LUND_DIR, f"{stem}.png")
        panel_events = phys_events[panel_mask]
        plot_lund_plane(panel_events, label=label, save_path=lund_path)
        # Paper figures are at repo root / paper / figures
        repo_root = os.path.dirname(os.path.dirname(model_dir))
        paper_path = os.path.join(repo_root, "paper", "figures",
                                  f"lund_subclass_{stem.replace('subclass_', '')}.png")
        copy2(lund_path, paper_path)
        results["panels"][stem] = {"subclass": SUBCLASS_NAMES[sc], "status": status_filter,
                                   "n_total": n_total, "n_plotted": n_panel, **panel_stats(panel_events)}
        print(f"  {stem}: n_total={n_total:>7d}  n_plotted={n_panel:>5d}  -> {os.path.basename(paper_path)}")

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUT_JSON}  [COMPLETE] ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()