"""Arc-length order profiles on the ArcFace prototype plane.

Tests whether subclasses form an ordered continuum along the QCD->Top
decision axis or merely compositional clusters.  Uses one deterministic
50k-event test subset, ArcFace only: project L2-normalized embeddings onto
the prototype decision axis and an orthogonal residual direction, then bin
by the decision coordinate and profile mass / tau32 / margin.

Outputs (experiments only):
- experiments/arc_order_profile.json
- experiments/arc_order_profile.png
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
from sklearn.decomposition import PCA
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
N_BINS = 25
SCATTER_MAX = 6000
SCATTER_SEED = 123


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


def monotone_stats(order, values):
    """Spearman rho and fraction of monotonic successive steps."""
    order = np.asarray(order, dtype=float)
    values = np.asarray(values, dtype=float)
    rho = float(np.corrcoef(rankdata(order), rankdata(values))[0, 1])
    steps = np.diff(values)
    if len(steps) == 0:
        return rho, 0.0
    # Sign is judged against the overall end-to-end direction.
    expected = np.sign(values[-1] - values[0])
    if expected == 0:
        return rho, 0.0
    frac = float(np.mean(np.sign(steps) == expected))
    return rho, frac


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
    acc = float(np.mean(preds == y_true))
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
    resid = emb - np.outer(emb @ e1, e1)
    pca_resid = PCA(n_components=1)
    pca_resid.fit(resid)
    e2 = pca_resid.components_[0]
    if (emb[preds == 1] @ e2).mean() < (emb[preds == 0] @ e2).mean():
        e2 = -e2
    u = emb @ e1
    v = emb @ e2

    sub = assign_predicted_subclasses(emb, preds)
    counts = {int(s): int((sub == s).sum()) for s in range(4)}
    cents = {s: np.array([u[sub == s].mean(), v[sub == s].mean()]) for s in range(4)}

    # Bin by the decision coordinate (arc-length proxy along QCD->Top).
    qs = np.quantile(u, np.linspace(0, 1, N_BINS + 1))
    qs[0] -= 1e-9
    qs[-1] += 1e-9
    bins = np.clip(np.digitize(u, qs) - 1, 0, N_BINS - 1)
    order, m_mass, m_tau, m_margin, m_n = [], [], [], [], []
    comp = []
    for b in range(N_BINS):
        m = bins == b
        if m.sum() < 50:
            continue
        order.append(b)
        m_mass.append(mass[m].mean())
        m_tau.append(tau32[m].mean())
        m_margin.append(margin[m].mean())
        m_n.append(int(m.sum()))
        comp.append({str(s): round(float((sub[m] == s).mean()), 4) for s in range(4)})
    order = np.array(order)
    stats = {}
    for key, vals in (("mass", m_mass), ("tau32", m_tau), ("margin", m_margin)):
        rho, frac = monotone_stats(order, np.array(vals))
        stats[key] = {"spearman": round(rho, 4), "monotone_frac": round(frac, 4)}
    print(f"acc={acc:.4f} " +
          " ".join(f"{k}: rho={v['spearman']}, mono={v['monotone_frac']}" for k, v in stats.items()))

    # Per-class ordering: QCD mass along s, Top tau32 along s.
    per_class = {}
    for cls, tkey, tvals in ((0, "mass", mass), (1, "tau32", tau32)):
        m = preds == cls
        uu, yy = u[m], tvals[m]
        idx = np.argsort(uu)
        uu, yy = uu[idx], yy[idx]
        edges = np.quantile(uu, np.linspace(0, 1, 13))
        bm, bv = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mm = (uu >= lo) & (uu <= hi)
            if mm.sum() >= 200:
                bm.append(uu[mm].mean())
                bv.append(yy[mm].mean())
        rho, frac = monotone_stats(np.arange(len(bv)), np.array(bv))
        per_class[f"pred{cls}_{tkey}"] = {
            "spearman": round(rho, 4), "monotone_frac": round(frac, 4),
            "n_bins": len(bv),
        }
        print(f"  pred{cls} {tkey}: rho={rho:.3f}, mono={frac:.3f}, bins={len(bv)}")

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [1.15, 1.0]})
    fig.subplots_adjust(wspace=0.28, left=0.07, right=0.97, top=0.88, bottom=0.16)
    rng = np.random.default_rng(123)
    idx = rng.choice(len(u), min(6000, len(u)), replace=False)
    cols = [SUBCLASS_COLORS["QCD core"], SUBCLASS_COLORS["QCD edge"],
            SUBCLASS_COLORS["Top core"], SUBCLASS_COLORS["Top edge"]]
    ax = axes[0]
    for sc, color in enumerate(cols):
        m = np.intersect1d(idx, np.where(sub == sc)[0])
        ax.scatter(u[m], v[m], s=6, alpha=0.35, c=color, edgecolors="none",
                   rasterized=True, zorder=2)
    for sc, color in enumerate(cols):
        ax.scatter(cents[sc][0], cents[sc][1], s=42, c=color,
                   edgecolors="black", linewidths=0.6, zorder=5)
    for a, b, style, width in (
        (cents[0], cents[1], "-", 1.5),
        (cents[2], cents[3], "-", 1.5),
        (cents[0], cents[2], "--", 1.1),
        (cents[1], cents[2], "-", 1.2),
        (cents[0], cents[3], ":", 1.1),
    ):
        ax.annotate("", xy=b, xytext=a,
                    arrowprops=dict(arrowstyle="-|>", color="black",
                                    lw=width, linestyle=style), zorder=6)
    ax.set_xlabel("prototype decision axis $u$ (QCD $\\to$ Top)")
    ax.set_ylabel("orthogonal residual $v$")
    ax.set_aspect("equal", adjustable="datalim")
    add_grid(ax)
    panel_label(ax, "a")
    ax.set_title("ArcFace prototype plane: endpoints + transitions", fontsize=9, pad=6)

    ax = axes[1]
    xx = np.array(order, dtype=float)
    ax.plot(xx, m_mass, "o-", color=COLORS["qcd"], label="mass [GeV]",
            markersize=3, linewidth=1.1)
    ax2 = ax.twinx()
    ax2.plot(xx, m_tau, "s-", color=COLORS["top"], label=r"$\tau_{32}$",
             markersize=3, linewidth=1.1)
    ax.set_xlabel("decision-order bin (QCD $\\to$ Top)")
    ax.set_ylabel("mean mass [GeV]")
    ax2.set_ylabel(r"mean $\tau_{32}$")
    add_grid(ax)
    panel_label(ax, "b")
    ax.set_title("Binned profiles along the decision axis", fontsize=9, pad=6)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=7)

    # Fix color refs (defined via figure_style COLORS).
    save_fig(fig, "arc_order_profile.png", to_paper=False)

    out = EXPERIMENTS / "arc_order_profile.json"
    payload = {
        "n_events": MAX_EVENTS,
        "accuracy": round(acc, 6),
        "n_bins": N_BINS,
        "min_bin_n": int(min(m_n)) if m_n else 0,
        "profile_stats": stats,
        "per_class": per_class,
        "subclass_counts": counts,
        "bin_composition": comp,
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved {out} ({payload['elapsed_s']}s)\n[DONE]")


if __name__ == "__main__":
    main()
