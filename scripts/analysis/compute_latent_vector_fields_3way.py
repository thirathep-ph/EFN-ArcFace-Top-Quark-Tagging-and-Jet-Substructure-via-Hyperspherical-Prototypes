"""Latent physics vector fields for the three EFN heads.

Uses one deterministic 50k-event test subset for ArcFace, CosLinear, and
Linear so the comparison is paired.  Embeddings are L2-normalized before a
per-head PCA-2 projection, matching the three-head PCA panels in docs/RESULTS.md.  At
each 2D grid cell, a local least-squares fit estimates the direction in
which mass, tau32, and decision margin increase.  The result tests whether
the manifold has coherent physical directions, not only clusters.
"""

import gc
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader

from arcefn.data.loader import load_awkward, JetTaggingDataset
from arcefn.utils.clustering import spectral_clustering_subclass
from arcefn.utils.device import get_device
from arcefn.utils.figure_style import (
    COLORS,
    SUBCLASS_COLORS,
    add_grid,
    apply_style,
    panel_label,
    save_fig,
)
from arcefn.utils.model_loading import load_model_from_dir
from arcefn.utils.paths import DATA_DIR, EXPERIMENTS
from arcefn.utils.physics import compute_features

MODEL_DIRS = {
    "ArcFace": EXPERIMENTS / "robustness_s16_m05",
    "CosLinear": EXPERIMENTS / "baselines_coslinear_seed_42",
    "Linear": EXPERIMENTS / "baselines_linear_seed_42",
}
LABELS = {
    "ArcFace": r"ArcFace ($S^{63},\,m=0.5$)",
    "CosLinear": r"CosLinear ($S^{63},\,m=0$)",
    "Linear": r"Linear (Euclid.)",
}
MAX_EVENTS = 50000
BATCH_SIZE = 1024
GRID_N = 13
K_NEIGHBORS = 320
SCATTER_SEED = 123
SCATTER_MAX = 4000
TARGETS = {"mass": 0, "tau32": 7}


def get_model_outputs(model_dir, loader, device):
    """Return raw embeddings, logits, and predictions."""
    model = load_model_from_dir(Path(model_dir), device)
    model.eval()
    embs, logits = [], []
    with torch.no_grad():
        for x, _, _, mask, _ in loader:
            out = model(x.to(device), mask=mask.to(device))
            embs.append(out[-1].cpu().numpy())
            logits.append(out[0].cpu().numpy())
    del model
    gc.collect()
    embs = np.concatenate(embs, axis=0)
    logits = np.concatenate(logits, axis=0)
    return embs, logits, np.argmax(logits, axis=1)


def assign_predicted_subclasses(emb_norm, preds):
    """Cluster each predicted class separately; name the larger Core."""
    sub = np.full(len(preds), -1, dtype=int)
    for cls, offset in ((0, 0), (1, 2)):
        mask = preds == cls
        if mask.sum() < 10:
            continue
        labels, _ = spectral_clustering_subclass(emb_norm[mask], max_k=5)
        counts = sorted(
            ((int(c), int((labels == c).sum())) for c in np.unique(labels) if c >= 0),
            key=lambda item: -item[1],
        )
        if len(counts) >= 2:
            mapping = {counts[0][0]: 0, counts[1][0]: 1}
            mapped = np.array([mapping.get(int(v), 0) for v in labels], dtype=int)
        else:
            mapped = np.zeros(mask.sum(), dtype=int)
        sub[mask] = mapped + offset
    return sub


def unit_vector(a, b):
    """Return the unit vector from centroid a to centroid b."""
    v = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    n = float(np.linalg.norm(v))
    return (v / n if n > 0 else v), n


def orient_axes(coords, mass, tau32):
    """Orient PC1 pro-mass and PC2 pro-tau32 for comparable panels."""
    flips = [1.0, 1.0]
    if np.corrcoef(coords[:, 0], mass)[0, 1] < 0:
        coords[:, 0] *= -1.0
        flips[0] = -1.0
    if np.corrcoef(coords[:, 1], tau32)[0, 1] < 0:
        coords[:, 1] *= -1.0
        flips[1] = -1.0
    return coords, flips


def local_vector_field(coords, target):
    """Estimate local target gradients on a uniform 2D grid."""
    lo = np.quantile(coords, 0.01, axis=0)
    hi = np.quantile(coords, 0.99, axis=0)
    gx = np.linspace(lo[0], hi[0], GRID_N)
    gy = np.linspace(lo[1], hi[1], GRID_N)
    centers = np.array(np.meshgrid(gx, gy)).reshape(2, -1).T

    tree = NearestNeighbors(n_neighbors=K_NEIGHBORS)
    tree.fit(coords)
    dist, idx = tree.kneighbors(centers)
    radius = np.quantile(dist[:, -1], 0.90)

    arrows, r2, support = [], [], []
    for center, neighbors, dmax in zip(centers, idx, dist[:, -1]):
        if dmax > radius:
            continue
        u = coords[neighbors] - center
        y = target[neighbors]
        if np.var(y) <= 0:
            continue
        a = np.column_stack([np.ones(len(u)), u])
        beta, *_ = np.linalg.lstsq(a, y, rcond=None)
        pred = a @ beta
        denom = float(np.sum((y - y.mean()) ** 2))
        if denom <= 0:
            continue
        r2.append(1.0 - float(np.sum((y - pred) ** 2)) / denom)
        g = beta[1:].astype(float)
        n = float(np.linalg.norm(g))
        if n > 0:
            arrows.append((center.copy(), g / n))
            support.append(len(u))
    return np.array(centers), arrows, np.array(r2), int(np.sum(support) / max(len(support), 1))


def summarize_direction(arrows):
    """Return the mean unit direction and directional concentration."""
    if not arrows:
        return None, 0.0
    vecs = np.array([v for _, v in arrows])
    mean = vecs.mean(axis=0)
    conc = float(np.linalg.norm(mean))
    unit = (mean / conc).tolist() if conc > 0 else [0.0, 0.0]
    return unit, conc


def main():
    t0 = time.time()
    device, _ = get_device()
    print(f"Device: {device}")

    print(f"Loading first {MAX_EVENTS:,} test events once (paired subset)...")
    events, y_true, _ = load_awkward(
        str(DATA_DIR / "test.h5"), max_events=MAX_EVENTS, lazy=False
    )
    print("Computing shared physics features once...")
    phys = compute_features(events)
    mass = phys[:, TARGETS["mass"]].astype(float)
    tau32 = phys[:, TARGETS["tau32"]].astype(float)

    loader = DataLoader(
        JetTaggingDataset(events, y_true, np.ones(len(y_true), dtype=np.float32)),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9), sharex=False, sharey=False)
    fig.subplots_adjust(wspace=0.28, left=0.06, right=0.98, top=0.86, bottom=0.18)
    results = {
        "n_events": MAX_EVENTS,
        "grid": GRID_N,
        "neighbors": K_NEIGHBORS,
        "scatter_max_per_head": SCATTER_MAX,
        "scatter_seed": SCATTER_SEED,
        "heads": {},
    }

    rng = np.random.default_rng(SCATTER_SEED)
    scatter_cols = [
        SUBCLASS_COLORS["QCD core"],
        SUBCLASS_COLORS["QCD edge"],
        SUBCLASS_COLORS["Top core"],
        SUBCLASS_COLORS["Top edge"],
    ]

    for col, (name, model_dir) in enumerate(MODEL_DIRS.items()):
        print(f"\n--- {name} ---")
        emb_raw, logits, preds = get_model_outputs(model_dir, loader, device)
        margin = (logits[:, 1] - logits[:, 0]).astype(float)
        acc = float(np.mean(preds == y_true))
        emb_norm = emb_raw / (np.linalg.norm(emb_raw, axis=1, keepdims=True) + 1e-12)

        pca = PCA(n_components=2)
        coords = pca.fit_transform(emb_norm)
        coords, flips = orient_axes(coords, mass, tau32)
        variance = [round(float(v), 4) for v in pca.explained_variance_ratio_]

        sub = assign_predicted_subclasses(emb_norm, preds)
        centroids, counts = {}, {}
        for sc in range(4):
            mask = sub == sc
            counts[sc] = int(mask.sum())
            centroids[sc] = coords[mask].mean(axis=0).tolist() if mask.sum() else [0.0, 0.0]
        qcd_dir, _ = unit_vector(centroids[0], centroids[1])
        top_dir, _ = unit_vector(centroids[2], centroids[3])
        class_dir, _ = unit_vector(
            np.mean([centroids[0], centroids[1]], axis=0),
            np.mean([centroids[2], centroids[3]], axis=0),
        )

        targets = {"mass": mass, "tau32": tau32, "margin": margin}
        fields = {}
        target_arrows = {}
        for key, values in targets.items():
            _, arrows, r2, _ = local_vector_field(coords, values)
            unit, conc = summarize_direction(arrows)
            target_arrows[key] = arrows
            fields[key] = {
                "n_cells": int(len(arrows)),
                "median_r2": round(float(np.median(r2)), 4) if len(r2) else None,
                "mean_unit": [round(float(v), 4) for v in unit] if unit else None,
                "directional_concentration": round(float(conc), 4),
                "cos_qcd_core_edge": round(float(np.dot(unit, qcd_dir)), 4) if unit else None,
                "cos_top_core_edge": round(float(np.dot(unit, top_dir)), 4) if unit else None,
                "cos_class_qcd_top": round(float(np.dot(unit, class_dir)), 4) if unit else None,
            }
            print(f"  {key}: cells={len(arrows)}, median R2={fields[key]['median_r2']}, "
                  f"conc={fields[key]['directional_concentration']}, "
                  f"cos(QCD)={fields[key]['cos_qcd_core_edge']}, "
                  f"cos(Top)={fields[key]['cos_top_core_edge']}")

        results["heads"][name] = {
            "model_dir": str(model_dir),
            "accuracy": round(acc, 6),
            "pca_variance": variance,
            "pca_cum2": round(float(sum(variance)), 4),
            "axis_flips": flips,
            "subclass_counts": counts,
            "fields": fields,
        }

        ax = axes[col]
        idx = rng.choice(len(coords), min(SCATTER_MAX, len(coords)), replace=False)
        ax.scatter(
            coords[idx, 0], coords[idx, 1], s=4, alpha=0.18, c="0.55",
            edgecolors="none", rasterized=True, zorder=1,
        )
        for sc, color in enumerate(scatter_cols):
            if counts[sc]:
                ax.scatter(
                    centroids[sc][0], centroids[sc][1], s=34, c=color,
                    edgecolors="black", linewidths=0.5, zorder=5,
                )
        ax.annotate(
            "", xy=centroids[1], xytext=centroids[0],
            arrowprops=dict(arrowstyle="-|>", color="black", lw=1.4), zorder=6,
        )
        ax.annotate(
            "", xy=centroids[3], xytext=centroids[2],
            arrowprops=dict(arrowstyle="-|>", color="black", lw=1.4), zorder=6,
        )
        for key, color in (("mass", COLORS["qcd"]), ("tau32", COLORS["top"])):
            arrows = target_arrows[key]
            if arrows:
                pts = np.array([p for p, _ in arrows])
                vec = np.array([v for _, v in arrows])
                ax.quiver(
                    pts[:, 0], pts[:, 1], vec[:, 0], vec[:, 1],
                    color=color, width=0.0065, headwidth=3.2, headlength=4.2,
                    alpha=0.82, zorder=4, label=key,
                )
        ax.set_xlabel("PC1 (mass-oriented)")
        ax.set_ylabel("PC2 ($\\tau_{32}$-oriented)")
        ax.set_aspect("equal", adjustable="datalim")
        add_grid(ax)
        panel_label(ax, chr(97 + col))
        ax.set_title(
            f"{LABELS[name]}\nPC1+PC2={100*sum(variance):.1f}%; "
            f"mass $R^2={fields['mass']['median_r2']:.2f}$, "
            f"$\\tau_{{32}}$ $R^2={fields['tau32']['median_r2']:.2f}$",
            fontsize=7, pad=6,
        )

    handles = [
        plt.Line2D([0], [0], color=COLORS["qcd"], lw=2, label="mass direction"),
        plt.Line2D([0], [0], color=COLORS["top"], lw=2, label="$\\tau_{32}$ direction"),
        plt.Line2D([0], [0], color="black", lw=2, label="Core to Edge"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=8,
               frameon=True, bbox_to_anchor=(0.5, 1.02))
    save_fig(fig, "latent_vector_field_3way.png")

    out = EXPERIMENTS / "latent_vector_fields_3way.json"
    results["elapsed_s"] = round(time.time() - t0, 1)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out} ({results['elapsed_s']}s)\n[DONE]")


if __name__ == "__main__":
    main()
