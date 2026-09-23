"""Latent physics gradient fields, one observable per panel, three heads.

This follows the requested ParT-style figure: each panel shows one physics
observable on a per-head PCA plane.  Scatter points are colored by the raw
observable value; arrows are local least-squares gradients on that plane and
are colored by field strength.  PCA axes are *not* pre-oriented to mass or
tau32; panels are oriented only by predicted class (Top to the right/upper)
so the figure cannot bake the answer into the frame.

Outputs:
- experiments/latent_vector_fields_parh_3x3.json
- experiments/latent_vector_field_parh_3x3.png (+ paper/figures copy)
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
from matplotlib.colors import Normalize
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
HEAD_LABELS = {
    "ArcFace": r"ArcFace ($S^{63},\,m=0.5$)",
    "CosLinear": r"CosLinear ($S^{63},\,m=0$)",
    "Linear": r"Linear (Euclid.)",
}
TARGETS = (
    ("mass", 0, "mass [GeV]", "viridis"),
    ("tau32", 7, r"$\tau_{32}$", "plasma"),
    ("margin", None, "logit difference", "cividis"),
)
MAX_EVENTS = 50000
BATCH_SIZE = 1024
GRID_N = 16
K_NEIGHBORS = 300
SCATTER_MAX = 6000
SCATTER_SEED = 123


def get_model_outputs(model_dir, loader, device):
    """Return raw embeddings, logits, and predicted labels."""
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


def orient_by_predicted_class(coords, preds):
    """Orient PC1/PC2 so predicted Top lies right/upper of QCD."""
    flips = [1.0, 1.0]
    for axis in (0, 1):
        if coords[preds == 1, axis].mean() < coords[preds == 0, axis].mean():
            coords[:, axis] *= -1.0
            flips[axis] = -1.0
    return coords, flips


def robust_standardize(values):
    """Center by median and scale by IQR for stable gradient magnitudes."""
    med = float(np.median(values))
    iqr = float(np.quantile(values, 0.75) - np.quantile(values, 0.25))
    if iqr <= 0:
        return values - med, med, 1.0
    return (values - med) / iqr, med, iqr


def local_vector_field(coords, target):
    """Estimate one grid of local gradients and their strength."""
    lo = np.quantile(coords, 0.02, axis=0)
    hi = np.quantile(coords, 0.98, axis=0)
    gx = np.linspace(lo[0], hi[0], GRID_N)
    gy = np.linspace(lo[1], hi[1], GRID_N)
    centers = np.array(np.meshgrid(gx, gy)).reshape(2, -1).T

    z, _, _ = robust_standardize(target.astype(float))
    tree = NearestNeighbors(n_neighbors=K_NEIGHBORS)
    tree.fit(coords)
    _, idx = tree.kneighbors(centers)

    pts, vecs, strengths, r2s = [], [], [], []
    for center, neighbors in zip(centers, idx):
        u = coords[neighbors] - center
        y = z[neighbors]
        if np.var(y) <= 0:
            continue
        a = np.column_stack([np.ones(len(u)), u])
        beta, *_ = np.linalg.lstsq(a, y, rcond=None)
        pred = a @ beta
        denom = float(np.sum((y - y.mean()) ** 2))
        if denom <= 0:
            continue
        r2 = 1.0 - float(np.sum((y - pred) ** 2)) / denom
        g = beta[1:].astype(float)
        n = float(np.linalg.norm(g))
        if n <= 0:
            continue
        pts.append(center)
        vecs.append(g / n)
        strengths.append(n)
        r2s.append(r2)
    return np.array(pts), np.array(vecs), np.array(strengths), np.array(r2s)


def unit_vector(a, b):
    """Return the unit vector from centroid a to centroid b."""
    v = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    n = float(np.linalg.norm(v))
    return (v / n if n > 0 else v), n


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
    loader = DataLoader(
        JetTaggingDataset(events, y_true, np.ones(len(y_true), dtype=np.float32)),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    apply_style()
    fig, axes = plt.subplots(3, 3, figsize=(11.5, 10.2), sharex="col")
    fig.subplots_adjust(wspace=0.22, hspace=0.34, left=0.07, right=0.90,
                        top=0.91, bottom=0.07)
    results = {
        "n_events": MAX_EVENTS,
        "grid": GRID_N,
        "neighbors": K_NEIGHBORS,
        "scatter_seed": SCATTER_SEED,
        "scatter_max_per_head": SCATTER_MAX,
        "frame": "per-head PCA-2 of L2-normalized embeddings; oriented only by predicted class",
        "heads": {},
    }
    rng = np.random.default_rng(SCATTER_SEED)
    scatter_cols = [
        SUBCLASS_COLORS["QCD core"],
        SUBCLASS_COLORS["QCD edge"],
        SUBCLASS_COLORS["Top core"],
        SUBCLASS_COLORS["Top edge"],
    ]

    # Precompute global color ranges per observable across heads.
    cached = {}
    for name, model_dir in MODEL_DIRS.items():
        emb_raw, logits, preds = get_model_outputs(model_dir, loader, device)
        emb_norm = emb_raw / (np.linalg.norm(emb_raw, axis=1, keepdims=True) + 1e-12)
        pca = PCA(n_components=2)
        coords = pca.fit_transform(emb_norm)
        coords, _ = orient_by_predicted_class(coords, preds)
        margin = (logits[:, 1] - logits[:, 0]).astype(float)
        sub = assign_predicted_subclasses(emb_norm, preds)
        cached[name] = {
            "coords": coords,
            "preds": preds,
            "margin": margin,
            "sub": sub,
            "logits_acc": round(float(np.mean(preds == y_true)), 6),
            "pca_variance": [round(float(v), 4) for v in pca.explained_variance_ratio_],
        }
    targets = {
        "mass": phys[:, 0].astype(float),
        "tau32": phys[:, 7].astype(float),
    }
    ranges = {}
    for key, idx, _, _ in TARGETS:
        vals = np.concatenate([
            targets[key] if key in targets else cached[h]["margin"]
            for h in MODEL_DIRS
        ]).astype(float)
        lo, hi = float(np.quantile(vals, 0.02)), float(np.quantile(vals, 0.98))
        ranges[key] = (lo, hi if hi > lo else lo + 1.0)

    for row, (name, _) in enumerate(MODEL_DIRS.items()):
        info = cached[name]
        coords = info["coords"]
        preds = info["preds"]
        sub = info["sub"]
        centroids = {}
        counts = {}
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
        results["heads"][name] = {
            "accuracy": info["logits_acc"],
            "pca_variance": info["pca_variance"],
            "pca_cum2": round(float(sum(info["pca_variance"])), 4),
            "subclass_counts": counts,
            "fields": {},
        }

        values_by_target = {
            "mass": targets["mass"],
            "tau32": targets["tau32"],
            "margin": info["margin"],
        }
        for col, (key, _, color_label, cmap) in enumerate(TARGETS):
            ax = axes[row, col]
            values = values_by_target[key]
            pts, vecs, strengths, r2s = local_vector_field(coords, values)

            lo, hi = ranges[key]
            idx = rng.choice(len(coords), min(SCATTER_MAX, len(coords)), replace=False)
            sc = ax.scatter(
                coords[idx, 0], coords[idx, 1], c=values[idx], s=5, alpha=0.42,
                cmap=cmap, vmin=lo, vmax=hi, edgecolors="none",
                rasterized=True, zorder=1,
            )
            for sc_id, color in enumerate(scatter_cols):
                if counts[sc_id]:
                    ax.scatter(
                        centroids[sc_id][0], centroids[sc_id][1], s=30, c=color,
                        edgecolors="black", linewidths=0.5, zorder=6,
                    )
            ax.annotate("", xy=centroids[1], xytext=centroids[0],
                        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.3), zorder=7)
            ax.annotate("", xy=centroids[3], xytext=centroids[2],
                        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.3), zorder=7)

            if len(pts):
                span = max(
                    float(np.ptp(coords[:, 0])),
                    float(np.ptp(coords[:, 1])),
                    1e-6,
                )
                arrow_len = 0.055 * span
                ux = np.array([p for p in pts])
                uv = np.array([v for v in vecs]) * arrow_len
                s_lo, s_hi = float(np.quantile(strengths, 0.05)), float(np.quantile(strengths, 0.95))
                norm = Normalize(vmin=s_lo, vmax=s_hi if s_hi > s_lo else s_lo + 1.0)
                ax.quiver(
                    ux[:, 0], ux[:, 1], uv[:, 0], uv[:, 1], strengths,
                    cmap="coolwarm", norm=norm, angles="xy", scale_units="xy",
                    scale=1.0, width=0.006, headwidth=3.4, headlength=4.4,
                    alpha=0.88, zorder=4,
                )
                unit = vecs.mean(axis=0)
                conc = float(np.linalg.norm(unit))
                mean_unit = (unit / conc).tolist() if conc > 0 else [0.0, 0.0]
            else:
                mean_unit, conc = [0.0, 0.0], 0.0
            results["heads"][name]["fields"][key] = {
                "n_cells": int(len(pts)),
                "median_r2": round(float(np.median(r2s)), 4) if len(r2s) else None,
                "mean_unit": [round(float(v), 4) for v in mean_unit],
                "directional_concentration": round(float(conc), 4),
                "cos_qcd_core_edge": round(float(np.dot(mean_unit, qcd_dir)), 4),
                "cos_top_core_edge": round(float(np.dot(mean_unit, top_dir)), 4),
                "cos_class_qcd_top": round(float(np.dot(mean_unit, class_dir)), 4),
            }

            ax.set_aspect("equal", adjustable="datalim")
            add_grid(ax)
            if row == 0:
                ax.set_title(f"{color_label}", fontsize=9, pad=8)
            if col == 0:
                ax.set_ylabel(f"{HEAD_LABELS[name]}\nPC2")
            else:
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)
            if row == 2:
                ax.set_xlabel("PC1")
            else:
                ax.tick_params(labelbottom=False)
            panel_label(ax, f"{chr(97 + row * 3 + col)}")
            ax.text(
                0.02, 0.97,
                f"PC1+PC2={100*sum(info['pca_variance']):.1f}%; "
                f"$R^2={results['heads'][name]['fields'][key]['median_r2']:.2f}$",
                transform=ax.transAxes, ha="left", va="top", fontsize=6.5,
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="0.65", alpha=0.95),
            )

        print(f"{name}: acc={info['logits_acc']:.4f}, "
              f"PC cum2={sum(info['pca_variance']):.3f}, counts={counts}")

    import matplotlib.cm as cm
    for col, (key, _, color_label, cmap_name) in enumerate(TARGETS):
        lo, hi = ranges[key]
        norm = Normalize(vmin=lo, vmax=hi)
        mappable = cm.ScalarMappable(norm=norm, cmap=cmap_name)
        # One shared vertical bar per observable column.
        cbar_ax = fig.add_axes((0.908 + 0.027 * col, 0.07, 0.016, 0.84))
        cbar = fig.colorbar(mappable, cax=cbar_ax)
        cbar.set_label(color_label, fontsize=7)
        cbar.ax.tick_params(labelsize=6)

    save_fig(fig, "latent_vector_field_parh_3x3.png")

    out = EXPERIMENTS / "latent_vector_fields_parh_3x3.json"
    results["elapsed_s"] = round(time.time() - t0, 1)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out} ({results['elapsed_s']}s)\n[DONE]")


if __name__ == "__main__":
    main()
