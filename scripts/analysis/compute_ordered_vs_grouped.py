"""Ordered continuum vs compositional groups test (ArcFace, paired 50k).

Three decisive checks on the prototype decision axis u (QCD->Top):
1. Density/gap profile: equal-width histogram along u; are there valleys
   separating the four subclasses, or a filled continuum?
2. Within-subclass slopes: regress mass (QCD subclasses) and tau32
   (Top subclasses) on u *within* each subclass. Continuous -> same-sign
   slopes continue the global trend; compositional -> flat within groups
   with jumps at boundaries.
3. Mixed-bin mixture test: in boundary regions where subclass labels mix,
   fit 1- vs 2-component GMM on the relevant observable. Intermediate
   unimodal -> continuum; two components matching neighbors -> boundary.

Outputs (experiments only):
- experiments/ordered_vs_grouped.json
- experiments/ordered_vs_grouped.png
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
from scipy.stats import rankdata
from sklearn.mixture import GaussianMixture
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

MODEL_DIR = EXPERIMENTS / "robustness_s16_m05"
MAX_EVENTS = 50000
BATCH_SIZE = 1024
N_DENSITY_BINS = 60
GMM_SEED = 0


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


def slope_stats(x, y):
    """Slope, R2, Spearman rho of y on x."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.var(x) <= 0 or np.var(y) <= 0 or len(x) < 10:
        return {"slope": 0.0, "r2": 0.0, "spearman": 0.0, "n": int(len(x))}
    a = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(a, y, rcond=None)
    pred = a @ beta
    denom = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum((y - pred) ** 2)) / denom if denom > 0 else 0.0
    rho = float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])
    return {"slope": round(float(beta[1]), 6), "r2": round(r2, 4),
            "spearman": round(rho, 4), "n": int(len(x))}


def gmm_bic(values, seed=GMM_SEED):
    """BIC for 1- vs 2-component GMM; positive delta favors 2 components."""
    v = np.asarray(values, dtype=float).reshape(-1, 1)
    v = v[np.isfinite(v).ravel()]
    if len(v) < 100:
        return None
    out = {}
    for k in (1, 2):
        gm = GaussianMixture(n_components=k, covariance_type="full",
                             n_init=5, random_state=seed)
        gm.fit(v)
        out[k] = {
            "bic": round(float(gm.bic(v)), 2),
            "means": [round(float(m[0]), 4) for m in gm.means_],
            "weights": [round(float(w), 4) for w in gm.weights_],
        }
    out["delta_bic_2minus1"] = round(out[2]["bic"] - out[1]["bic"], 2)
    return out


def main():
    t0 = time.time()
    device, _ = get_device()
    print(f"Device: {device}")

    print(f"Loading first {MAX_EVENTS:,} test events...")
    events, y_true, _ = load_awkward(
        str(DATA_DIR / "test.h5"), max_events=MAX_EVENTS, lazy=True
    )
    phys = np.load(EXPERIMENTS / "rf_cache_test.npz")
    assert np.array_equal(y_true[:MAX_EVENTS], phys["labels"][:MAX_EVENTS]), \
        "physics cache order does not match test.h5 subset"
    feats = phys["features"][:MAX_EVENTS].astype(float)
    mass, tau32 = feats[:, 0], feats[:, 7]

    loader = DataLoader(
        JetTaggingDataset(events, y_true, np.ones(len(y_true), dtype=np.float32)),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )
    model = load_model_from_dir(Path(MODEL_DIR), device)
    model.eval()
    embs, logits = [], []
    with torch.no_grad():
        for x, _, _, mask, _ in loader:
            out = model(x.to(device), mask=mask.to(device))
            embs.append(out[-1].cpu().numpy())
            logits.append(out[0].cpu().numpy())
    del model
    gc.collect()
    emb_raw = np.concatenate(embs, axis=0)
    logits = np.concatenate(logits, axis=0)
    preds = np.argmax(logits, axis=1)
    margin = (logits[:, 1] - logits[:, 0]).astype(float)
    emb = emb_raw / (np.linalg.norm(emb_raw, axis=1, keepdims=True) + 1e-12)

    # Prototype decision axis from learned class centers.
    sd = torch.load(str(Path(MODEL_DIR) / "checkpoint.pt"),
                    map_location="cpu", weights_only=False)
    centers = sd["arcface_head.class_centers"]
    if torch.is_tensor(centers):
        centers = centers.detach().cpu().numpy()
    centers = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12)
    e1 = centers[1] - centers[0]
    e1 /= np.linalg.norm(e1) + 1e-12
    u = emb @ e1

    sub = assign_predicted_subclasses(emb, preds)

    results = {"n_events": MAX_EVENTS, "accuracy": round(float(np.mean(preds == y_true)), 6)}

    # --- 1. Density/gap profile along u (equal-width bins) ---
    hist, edges = np.histogram(u, bins=N_DENSITY_BINS)
    bc = (edges[:-1] + edges[1:]) / 2
    # Local minima strictly inside the support, relative to neighbors.
    valleys = []
    for i in range(1, len(hist) - 1):
        if hist[i] < hist[i - 1] and hist[i] < hist[i + 1] and hist[i] < 0.5 * max(hist[i - 1], hist[i + 1]):
            valleys.append({"bin": i, "center": round(float(bc[i]), 4),
                            "count": int(hist[i]),
                            "neighbors": [int(hist[i - 1]), int(hist[i + 1])]})
    results["density"] = {
        "n_bins": N_DENSITY_BINS, "min_count": int(hist.min()),
        "max_count": int(hist.max()), "n_empty": int((hist == 0).sum()),
        "valleys": valleys,
    }
    print(f"density: min={hist.min()}, max={hist.max()}, empty={int((hist == 0).sum())}, "
          f"valleys={len(valleys)}")

    # --- 2. Within-subclass slopes ---
    slopes = {}
    slopes["QCD-Core_mass"] = slope_stats(u[sub == 0], mass[sub == 0])
    slopes["QCD-Edge_mass"] = slope_stats(u[sub == 1], mass[sub == 1])
    slopes["Top-Core_tau32"] = slope_stats(u[sub == 2], tau32[sub == 2])
    slopes["Top-Edge_tau32"] = slope_stats(u[sub == 3], tau32[sub == 3])
    slopes["QCD_all_mass"] = slope_stats(u[preds == 0], mass[preds == 0])
    slopes["Top_all_tau32"] = slope_stats(u[preds == 1], tau32[preds == 1])
    slopes["all_margin"] = slope_stats(u, margin)
    results["within_slopes"] = slopes
    for k, v in slopes.items():
        print(f"  {k}: slope={v['slope']}, r2={v['r2']}, rho={v['spearman']}, n={v['n']}")

    # --- 3. Mixed-bin mixture test ---
    # Boundary regions: QCD-Core/Edge interface and Top-Core/Edge interface,
    # defined by u-quantiles where subclass composition changes.
    order = np.argsort(u)
    uso, subo = u[order], sub[order]
    # QCD boundary: last pure-Core quantile to first pure-Edge quantile.
    qcd_core_u = uso[subo == 0]
    qcd_edge_u = uso[subo == 1]
    top_core_u = uso[subo == 2]
    top_edge_u = uso[subo == 3]
    mixtures = {}
    # QCD interface: overlap band between Core max-decile and Edge min-decile.
    lo, hi = np.quantile(qcd_core_u, 0.9), np.quantile(qcd_edge_u, 0.1)
    band = (u >= min(lo, hi)) & (u <= max(lo, hi))
    mixtures["QCD_interface_mass"] = gmm_bic(mass[band & (preds == 0)])
    mixtures["QCD_interface_n"] = int((band & (preds == 0)).sum())
    lo, hi = np.quantile(top_core_u, 0.9), np.quantile(top_edge_u, 0.1)
    band = (u >= min(lo, hi)) & (u <= max(lo, hi))
    mixtures["Top_interface_tau32"] = gmm_bic(tau32[band & (preds == 1)])
    mixtures["Top_interface_n"] = int((band & (preds == 1)).sum())
    results["mixtures"] = mixtures
    for k, v in mixtures.items():
        if isinstance(v, dict) and "delta_bic_2minus1" in v:
            print(f"  {k}: dBIC(2-1)={v['delta_bic_2minus1']}, "
                  f"means1={v[1]['means']}, means2={v[2]['means']}, "
                  f"w2={v[2]['weights']}")
        else:
            print(f"  {k}: {v}")

    # --- Figure: density + subclass strips + within-subclass trends ---
    apply_style()
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.4), sharex=True,
                             gridspec_kw={"height_ratios": [1.2, 1.0]})
    fig.subplots_adjust(hspace=0.18, left=0.08, right=0.97, top=0.93, bottom=0.10)
    ax = axes[0]
    ax.hist(u, bins=N_DENSITY_BINS, color="0.55", alpha=0.55, edgecolor="white",
            linewidth=0.4, rasterized=True)
    for v in valleys:
        ax.axvline(v["center"], color="red", ls="--", lw=1.0, alpha=0.8)
    ax.set_ylabel("count")
    add_grid(ax, axis="y")
    panel_label(ax, "a")
    ax.set_title("Density along the prototype decision axis (equal-width bins); "
                 "red dashes = valleys", fontsize=9, pad=6)

    ax = axes[1]
    cols = {"QCD-Core": "#377eb8", "QCD-Edge": "#9ecae1",
            "Top-Core": "#e41a1c", "Top-Edge": "#fdae6b"}
    # Subclass strips: rug of u per subclass at fixed y.
    for sc, (name, color) in enumerate(
        [("QCD-Core", cols["QCD-Core"]), ("QCD-Edge", cols["QCD-Edge"]),
         ("Top-Core", cols["Top-Core"]), ("Top-Edge", cols["Top-Edge"])]):
        m = sub == sc
        ax.scatter(u[m], np.full(m.sum(), 0.02 + 0.03 * sc), s=3, alpha=0.25,
                   c=color, edgecolors="none", rasterized=True)
    # Within-subclass trend lines: mass for QCD, tau32 for Top (twin axes).
    for sc, key, color in ((0, "mass", cols["QCD-Core"]), (1, "mass", cols["QCD-Edge"])):
        m = sub == sc
        uu = u[m]
        edges = np.quantile(uu, np.linspace(0, 1, 9))
        bx, by = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mm = (uu >= lo) & (uu <= hi)
            if mm.sum() >= 100:
                bx.append(uu[mm].mean())
                by.append(mass[m][mm].mean())
        ax.plot(bx, np.array(by) / 300.0, "o-", color=color, ms=3, lw=1.2)
    ax.set_xlabel("prototype decision axis $u$ (QCD $\\to$ Top)")
    ax.set_ylabel("mass / 300 [GeV]  +  subclass strips")
    add_grid(ax)
    panel_label(ax, "b")
    ax.set_title("Within-subclass trends (mass rescaled) + subclass strips", fontsize=9, pad=6)

    save_fig(fig, "ordered_vs_grouped.png", to_paper=False)

    out = EXPERIMENTS / "ordered_vs_grouped.json"
    results["elapsed_s"] = round(time.time() - t0, 1)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out}\n[DONE]")


if __name__ == "__main__":
    main()
