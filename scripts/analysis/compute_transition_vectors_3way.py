"""Transition vectors between classes and subclasses, three heads.

Paired 50k-event test subset.  For each head this computes, in the model's
embedding geometry:
- centroid displacement vectors for class and subclass transitions;
- Fisher/LDA-style discriminant directions with bootstrap CIs;
- TCAV-like concept sensitivity: how the model margin changes when moving
  source samples along each transition direction, versus random directions;
- physics differences (mass/tau32 means and Cohen's d) on the same subset.

Transitions:
- QCD->Top (predicted class centroids)
- QCD Core->Edge, Top Core->Edge (predicted subclasses)
- QCD-Edge->Top-Core, QCD-Core->Top-Edge (cross-class steps)

Outputs (experiments only):
- experiments/transition_vectors_3way.json
- experiments/transition_vectors_3way.png
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
from torch.utils.data import DataLoader

from arcefn.data.loader import load_awkward, JetTaggingDataset
from arcefn.utils.clustering import spectral_clustering_subclass
from arcefn.utils.device import get_device
from arcefn.utils.figure_style import (
    SUBCLASS_COLORS,
    add_grid,
    apply_style,
    panel_label,
    save_fig,
)
from arcefn.utils.model_loading import load_model_from_dir
from arcefn.utils.paths import DATA_DIR, EXPERIMENTS

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
MAX_EVENTS = 50000
BATCH_SIZE = 1024
BOOT = 200
RANDOM_DIRS = 200
SCATTER_MAX = 6000
SCATTER_SEED = 123
RNG = np.random.default_rng(7)


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


def centroid(X, mask):
    """Return the mean vector of selected rows."""
    return X[mask].mean(axis=0)


def unit(v):
    """Return a unit vector (or zeros if degenerate)."""
    n = float(np.linalg.norm(v))
    return (v / n if n > 0 else v), n


def fisher_direction(Xa, Xb, reg=1e-3):
    """Fisher/LDA direction between two groups with regularized covariance."""
    xa = np.asarray(Xa, dtype=float)
    xb = np.asarray(Xb, dtype=float)
    mu_a = xa.mean(axis=0)
    mu_b = xb.mean(axis=0)
    ca = np.cov(xa, rowvar=False) if len(xa) > 1 else np.zeros((xa.shape[1], xa.shape[1]))
    cb = np.cov(xb, rowvar=False) if len(xb) > 1 else np.zeros((xb.shape[1], xb.shape[1]))
    pooled = (ca * len(xa) + cb * len(xb)) / max(len(xa) + len(xb), 1)
    trace = float(np.trace(pooled)) / pooled.shape[0] if pooled.size else 1.0
    mat = pooled + (reg * max(trace, 1e-12)) * np.eye(pooled.shape[0])
    w = np.linalg.solve(mat, mu_b - mu_a)
    return unit(w)


def bootstrap_cosine(Xa, Xb, ref, n_boot=BOOT, seed=11):
    """Bootstrap CI for cosine between centroid displacement and reference."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        ia = rng.integers(0, len(Xa), len(Xa))
        ib = rng.integers(0, len(Xb), len(Xb))
        v, _ = unit(Xb[ib].mean(axis=0) - Xa[ia].mean(axis=0))
        vals.append(float(np.dot(v, ref)))
    vals = np.array(vals)
    return round(float(vals.mean()), 4), [
        round(float(np.quantile(vals, 0.025)), 4),
        round(float(np.quantile(vals, 0.975)), 4),
    ]


def concept_sensitivity(X_src, margin_src, direction, n_random=RANDOM_DIRS, seed=23):
    """Slope of model margin along a transition direction vs random null."""
    direction = np.asarray(direction, dtype=float)
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    s = (X_src - X_src.mean(axis=0)) @ direction
    slope, r2, corr = _slope_stats(s, margin_src)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_random):
        r = rng.normal(size=X_src.shape[1])
        r /= np.linalg.norm(r) + 1e-12
        rs = (X_src - X_src.mean(axis=0)) @ r
        rslope, _, _ = _slope_stats(rs, margin_src)
        null.append(rslope)
    null = np.array(null)
    pct = float(np.mean(null <= slope))
    return {
        "slope": round(float(slope), 6),
        "r2": round(float(r2), 4),
        "corr": round(float(corr), 4),
        "null_median_slope": round(float(np.median(null)), 6),
        "null_p95_slope": round(float(np.quantile(null, 0.95)), 6),
        "slope_percentile_vs_random": round(pct, 4),
    }


def _slope_stats(s, y):
    """Univariate slope, R2, and Pearson correlation."""
    s = np.asarray(s, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.var(s) <= 0 or np.var(y) <= 0:
        return 0.0, 0.0, 0.0
    a = np.column_stack([np.ones(len(s)), s])
    beta, *_ = np.linalg.lstsq(a, y, rcond=None)
    pred = a @ beta
    denom = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum((y - pred) ** 2)) / denom if denom > 0 else 0.0
    corr = float(np.corrcoef(s, y)[0, 1])
    return float(beta[1]), r2, corr


def cohen_d(a, b):
    """Cohen's d for two samples."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = float(np.sqrt(a.var() + b.var()) + 1e-12)
    return float((b.mean() - a.mean()) / denom) if denom > 0 else 0.0


def orient_by_predicted_class(coords, preds):
    """Orient PC axes so predicted Top lies right/upper of QCD."""
    for axis in (0, 1):
        if coords[preds == 1, axis].mean() < coords[preds == 0, axis].mean():
            coords[:, axis] *= -1.0
    return coords


def main():
    t0 = time.time()
    device, _ = get_device()
    print(f"Device: {device}")

    print(f"Loading first {MAX_EVENTS:,} test events once (paired subset)...")
    events, y_true, _ = load_awkward(
        str(DATA_DIR / "test.h5"), max_events=MAX_EVENTS, lazy=True
    )
    phys_data = np.load(EXPERIMENTS / "rf_cache_test.npz")
    assert np.array_equal(y_true[:MAX_EVENTS], phys_data["labels"][:MAX_EVENTS]), \
        "physics cache order does not match test.h5 subset"
    phys = phys_data["features"][:MAX_EVENTS].astype(float)
    mass, tau32 = phys[:, 0], phys[:, 7]
    loader = DataLoader(
        JetTaggingDataset(events, y_true, np.ones(len(y_true), dtype=np.float32)),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    transitions = [
        ("QCD->Top", lambda sub, pred: (pred == 0, pred == 1)),
        ("QCD Core->Edge", lambda sub, pred: (sub == 0, sub == 1)),
        ("Top Core->Edge", lambda sub, pred: (sub == 2, sub == 3)),
        ("QCD-Edge->Top-Core", lambda sub, pred: (sub == 1, sub == 2)),
        ("QCD-Core->Top-Edge", lambda sub, pred: (sub == 0, sub == 3)),
    ]

    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9), sharex=False, sharey=False)
    fig.subplots_adjust(wspace=0.26, left=0.06, right=0.98, top=0.86, bottom=0.18)
    results = {"n_events": MAX_EVENTS, "bootstrap": BOOT, "heads": {}}
    rng = np.random.default_rng(SCATTER_SEED)
    scatter_cols = ["#377eb8", "#9ecae1", "#e41a1c", "#fdae6b"]

    for col, (name, model_dir) in enumerate(MODEL_DIRS.items()):
        print(f"\n--- {name} ---")
        emb_raw, logits, preds = get_model_outputs(model_dir, loader, device)
        margin = (logits[:, 1] - logits[:, 0]).astype(float)
        acc = float(np.mean(preds == y_true))
        emb_norm = emb_raw / (np.linalg.norm(emb_raw, axis=1, keepdims=True) + 1e-12)
        sub = assign_predicted_subclasses(emb_norm, preds)

        pca = PCA(n_components=2)
        coords = pca.fit_transform(emb_norm)
        coords = orient_by_predicted_class(coords, preds)
        variance = [round(float(v), 4) for v in pca.explained_variance_ratio_]

        head = {
            "accuracy": round(acc, 6),
            "pca_variance": variance,
            "pca_cum2": round(float(sum(variance)), 4),
            "transitions": {},
        }
        print(f"  acc={acc:.4f}, PC cum2={sum(variance):.3f}")

        panel_lines = []
        for tname, sel in transitions:
            ma, mb = sel(sub, preds)
            xa, xb = emb_norm[ma], emb_norm[mb]
            if ma.sum() < 10 or mb.sum() < 10:
                continue
            disp, disp_len = unit(xb.mean(axis=0) - xa.mean(axis=0))
            fisher, _ = fisher_direction(xa, xb)
            cos_fisher = round(float(np.dot(disp, fisher)), 4)
            mean_cos, ci = bootstrap_cosine(xa, xb, disp)
            sens = concept_sensitivity(xa, margin[ma], disp)
            phys_block = {
                "n_source": int(ma.sum()),
                "n_target": int(mb.sum()),
                "delta_mass": round(float(mass[mb].mean() - mass[ma].mean()), 4),
                "delta_tau32": round(float(tau32[mb].mean() - tau32[ma].mean()), 4),
                "cohen_mass": round(cohen_d(mass[ma], mass[mb]), 4),
                "cohen_tau32": round(cohen_d(tau32[ma], tau32[mb]), 4),
            }
            head["transitions"][tname] = {
                "displacement_norm": round(float(disp_len), 6),
                "cos_centroid_fisher": cos_fisher,
                "centroid_cos_mean": mean_cos,
                "centroid_cos_ci95": ci,
                "margin_sensitivity": sens,
                "physics": phys_block,
            }
            print(f"  {tname}: n={ma.sum()}/{mb.sum()}, cos(centroid,Fisher)={cos_fisher}, "
                  f"slope pct={sens['slope_percentile_vs_random']}, "
                  f"d_mass={phys_block['cohen_mass']}, d_tau32={phys_block['cohen_tau32']}")
            panel_lines.append((tname, xa, xb))

        results["heads"][name] = head

        ax = axes[col]
        idx = rng.choice(len(coords), min(SCATTER_MAX, len(coords)), replace=False)
        ax.scatter(coords[idx, 0], coords[idx, 1], s=5, alpha=0.22, c="0.55",
                   edgecolors="none", rasterized=True, zorder=1)
        cents = {}
        for sc, color in enumerate(scatter_cols):
            mask = sub == sc
            if mask.sum():
                cents[sc] = coords[mask].mean(axis=0)
                ax.scatter(cents[sc][0], cents[sc][1], s=34, c=color,
                           edgecolors="black", linewidths=0.5, zorder=5)
        order = [(0, 1, "-", 1.5), (2, 3, "-", 1.5), (0, 2, "--", 1.0),
                 (1, 2, "-", 1.0), (0, 3, ":", 1.0)]
        # Predicted-class path uses averaged subclass endpoints.
        paths = [
            ("QCD->Top", (cents.get(0, cents.get(1)), cents.get(2, cents.get(3))), "-", 1.6),
            ("QCD Core->Edge", (cents.get(0), cents.get(1)), "-", 1.4),
            ("Top Core->Edge", (cents.get(2), cents.get(3)), "-", 1.4),
            ("QCD-Edge->Top-Core", (cents.get(1), cents.get(2)), "--", 1.1),
            ("QCD-Core->Top-Edge", (cents.get(0), cents.get(3)), ":", 1.1),
        ]
        for label, (a, b), style, width in paths:
            if a is None or b is None:
                continue
            ax.annotate("", xy=b, xytext=a,
                        arrowprops=dict(arrowstyle="-|>", color="black",
                                        lw=width, linestyle=style), zorder=6)
        ax.set_xlabel("PC1 (predicted-class oriented)")
        ax.set_ylabel("PC2 (predicted-class oriented)" if col == 0 else "")
        if col:
            ax.tick_params(labelleft=False)
        ax.set_aspect("equal", adjustable="datalim")
        add_grid(ax)
        panel_label(ax, chr(97 + col))
        ax.set_title(f"{name}\nPC1+PC2={100*sum(variance):.1f}%", fontsize=8, pad=6)

    save_fig(fig, "transition_vectors_3way.png")
    out = EXPERIMENTS / "transition_vectors_3way.json"
    results["elapsed_s"] = round(time.time() - t0, 1)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out} ({results['elapsed_s']}s)\n[DONE]")


if __name__ == "__main__":
    main()
