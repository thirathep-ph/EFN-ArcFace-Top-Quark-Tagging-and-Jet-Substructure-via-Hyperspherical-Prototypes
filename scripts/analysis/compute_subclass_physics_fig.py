"""Regenerate subclass_physics_distributions.png on the PREDICTED partition.

Layout mirrors the original figure: rows = mass, tau32, sqrt(d12);
cols = (filled) discovered subclasses per predicted class, TN-vs-FP
(QCD status split), TP-vs-FN (top status split). All from cached
artifacts (embeddings.npz + rf_cache_test.npz); no model forward pass.
"""
import json
import numpy as np
from pathlib import Path
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from arcefn.utils.clustering import spectral_clustering_subclass
from arcefn.utils.figure_style import (apply_style, save_fig, panel_label,
                                         add_grid, SUBCLASS_COLORS, LEGEND_KWARGS)
from arcefn.utils.paths import EXPERIMENTS
from arcefn.utils.pred_labels import arcface_preds_from_centers, check_cm

apply_style()
np.random.seed(42)

npz = np.load(EXPERIMENTS / "robustness_s16_m05" / "embeddings.npz")
E, y = npz["embeddings"], npz["labels"]
feat = np.load(EXPERIMENTS / "rf_cache_test.npz")["features"]
assert len(E) == len(y) == len(feat) == 404000

preds = arcface_preds_from_centers(E, EXPERIMENTS / "robustness_s16_m05" / "checkpoint.pt")
check_cm(preds, y, "physics_fig")
En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-10)

# predicted subclasses, larger cluster -> Core
sub = np.full(len(y), -1, dtype=int)
info = {}
for cls, offset in [(0, 0), (1, 2)]:
    mask = preds == cls
    sl, k = spectral_clustering_subclass(En[mask], max_k=5)
    cnt = Counter(sl)
    order = sorted(cnt.keys(), key=lambda c: -cnt[c])
    mapped = np.array([order.index(v) for v in sl])
    sub[mask] = mapped + offset
    info[cls] = {"k": int(k), "counts": [int((mapped == i).sum()) for i in range(2)]}
print("partition:", info)

tp = (preds == 1) & (y == 1)
fp = (preds == 1) & (y == 0)
tn = (preds == 0) & (y == 0)
fn = (preds == 0) & (y == 1)

mass, tau32 = feat[:, 0], feat[:, 7]
d12 = np.sqrt(np.clip(feat[:, 4], 0, None))

fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=False)
panels = [
    ("Mass [GeV]", mass, np.linspace(0, 300, 60),
     [(sub == 0, "QCD-Core", SUBCLASS_COLORS["QCD core"]),
      (sub == 1, "QCD-Edge", SUBCLASS_COLORS["QCD edge"])]),
    ("Mass [GeV]", mass, np.linspace(0, 300, 60),
     [(sub == 2, "Top-Core", SUBCLASS_COLORS["Top core"]),
      (sub == 3, "Top-Edge", SUBCLASS_COLORS["Top edge"])]),
    ("Tau32", tau32, np.linspace(0.2, 1.0, 60),
     [(sub == 0, "QCD-Core", SUBCLASS_COLORS["QCD core"]),
      (sub == 1, "QCD-Edge", SUBCLASS_COLORS["QCD edge"])]),
    ("Tau32", tau32, np.linspace(0.2, 1.0, 60),
     [(sub == 2, "Top-Core", SUBCLASS_COLORS["Top core"]),
      (sub == 3, "Top-Edge", SUBCLASS_COLORS["Top edge"])]),
]

legend_loc = ["upper right", "upper right", "upper left", "upper left"]
for i, (ylabel, vals, bins, groups) in enumerate(panels):
    ax = axes[i // 2, i % 2]
    peak = 0.0
    for mask, lab, col in groups:
        v = vals[mask]
        n, _, _ = ax.hist(v, bins=bins, density=True, histtype="step",
                          color=col, label=f"{lab} (n={mask.sum():,})")
        peak = max(peak, float(np.max(n)))
    ax.set_ylim(0, peak * 1.3)
    panel_label(ax, "(%s)" % chr(97 + i))
    ax.set_ylabel(ylabel)
    ax.tick_params(labelsize=8)
    add_grid(ax)
    ax.legend(loc=legend_loc[i],
              **{k: v for k, v in LEGEND_KWARGS.items() if k != "fontsize"})

plt.tight_layout(pad=0.6, w_pad=1.2)
save_fig(fig, "subclass_physics_distributions.png")
print("saved subclass_physics_distributions.png")
plt.close(fig)
