"""Regenerate the subclass composition heatmap with canonical Core/Edge labels.

The paper's headline subclass partition clusters on PREDICTED labels over
the full 404k test set (what the model sees) and maps the largest cluster to
Core, the second to Edge.  Use --partition true to reproduce the GT check.
The composition heatmap must use the SAME partition and naming so its caption
is consistent with Table 2.  This script recomputes the per-subclass breakdown
by prediction status from the saved embeddings and cached class centers
(no forward pass, no UMAP).

Artifacts read:
  * experiments/robustness_s16_m05/embeddings.npz  -> 404k embeddings + labels
  * experiments/robustness_s16_m05/checkpoint.pt   -> class centers (predictions)

Output:
  * experiments/subclass_composition.json
  * paper/figures/subclass_composition_heatmap.png  (and experiments copy)
"""
from __future__ import annotations

import json
import time
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from arcefn.utils.clustering import spectral_clustering_subclass
from arcefn.utils.figure_style import apply_style, save_fig
from arcefn.utils.paths import EXPERIMENTS

EMBEDDINGS = EXPERIMENTS / "robustness_s16_m05" / "embeddings.npz"
CHECKPOINT = EXPERIMENTS / "robustness_s16_m05" / "checkpoint.pt"
OUT_JSON = EXPERIMENTS / "subclass_composition.json"
MAX_K = 5

SUBCLASS_NAMES = {0: "QCD-Core", 1: "QCD-Edge", 2: "Top-Core", 3: "Top-Edge"}
STATUS = ["True Positive", "False Positive", "True Negative", "False Negative"]


def normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-10)


def load_centers() -> np.ndarray:
    import torch
    try:
        import torch_directml  # noqa: F401  (registers PrivateUse1 so the checkpoint loads)
    except ImportError:
        pass
    sd = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    return normalize(sd["arcface_head.class_centers"].detach().cpu().numpy())


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", choices=["predicted", "true"], default="predicted",
                        help="predicted (model view) or true (GT consistency check)")
    args = parser.parse_args()
    t0 = time.time()
    emb_np = np.load(EMBEDDINGS)
    embeddings, labels = emb_np["embeddings"], emb_np["labels"]
    centers = load_centers()
    emb_norm = normalize(embeddings)
    preds = np.argmax(emb_norm @ centers.T, axis=1)
    partition_labels = preds if args.partition == "predicted" else labels

    true_global = np.full(len(labels), -1, dtype=int)
    for cls in (0, 1):
        mask = partition_labels == cls
        sub, _ = spectral_clustering_subclass(emb_norm[mask], max_k=MAX_K)
        counts = Counter(sub)
        order = sorted(counts, key=lambda c: -counts[c])  # largest -> Core (0)
        mapped = np.array([order.index(s) for s in sub])
        true_global[mask] = 2 * cls + mapped

    tp = (preds == 1) & (labels == 1)
    fp = (preds == 1) & (labels == 0)
    tn = (preds == 0) & (labels == 0)
    fn = (preds == 0) & (labels == 1)
    status = np.empty(len(labels), dtype=object)
    status[tp], status[fp], status[tn], status[fn] = STATUS

    composition: dict = {}
    matrix = []
    row_labels = []
    for sc in range(4):
        sc_mask = true_global == sc
        n_tot = int(sc_mask.sum())
        if n_tot == 0:
            continue
        row = []
        entry = {"subclass": SUBCLASS_NAMES[sc], "n": n_tot, "status_pct": {}}
        for cat in STATUS:
            pct = float(((status == cat) & sc_mask).sum() / n_tot * 100.0)
            row.append(pct)
            entry["status_pct"][cat] = pct
        composition[SUBCLASS_NAMES[sc]] = entry
        matrix.append(row)
        row_labels.append(f"{SUBCLASS_NAMES[sc]} (N={n_tot})")

    apply_style()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    mat = np.array(matrix)
    annot_labels = np.where(mat < 0.05, "", np.char.mod("%.1f", np.round(mat, 1)))
    sns.heatmap(mat, annot=annot_labels, fmt="", ax=ax,
                xticklabels=STATUS, yticklabels=row_labels,
                cmap="YlGnBu", cbar_kws={"label": "Percentage (%)", "shrink": 0.8},
                annot_kws={"fontsize": 8})
    ax.set_xlabel("Prediction status")
    ax.set_ylabel(f"Discovered subclass ({args.partition}-label partition)")
    ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()
    save_fig(fig, "subclass_composition_heatmap.png", to_paper=True)

    with open(OUT_JSON, "w") as f:
        json.dump({"n_events": int(len(labels)), "composition": composition}, f, indent=2)
    print(f"\nSaved: {OUT_JSON}  [COMPLETE] ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
