import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from arcefn.utils.figure_style import apply_style, panel_label, add_grid, save_fig, COLORS

inp = pathlib.Path("experiments/dim_3way_50k.json")
dat = json.loads(inp.read_text())

heads = ["ArcFace", "CosLinear", "Linear"]
labels = [r"ArcFace ($S^{63},\,m=0.5$)", r"CosLinear ($S^{63},\,m=0$)", r"Linear (Euclid.)"]
cols = [COLORS["top"], COLORS["purple"], COLORS["gray"]]

apply_style()
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
fig.subplots_adjust(wspace=0.22, left=0.07, right=0.98, top=0.88, bottom=0.20)

for ax, h, lab in zip(axes, heads, labels):
    pca = np.array(dat[h]["pca_top6"])
    x = np.arange(1, 7)
    ax.bar(x, pca, color=cols[heads.index(h)], edgecolor="black", linewidth=0.5, width=0.62)
    ax.set_xlabel("PC rank")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in x])
    ax.set_ylim(0, 0.92)
    ax.set_xlim(0.5, 6.5)
    add_grid(ax, axis="y")
    ax.set_title(f"({chr(97+heads.index(h))}) {lab}", fontsize=8, pad=6)
    tw = dat[h]["twonn"]
    pr = dat[h]["pr"]
    cum2 = dat[h]["pca_cum2"] * 100
    # top-right annotation without overlapping bars
    ax.text(0.97, 0.94, f"TwoNN {tw:.2f}\nPR {pr:.2f}\n2 PCs {cum2:.1f}%",
            transform=ax.transAxes, ha="right", va="top", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="0.65", alpha=0.95))
    for i, v in enumerate(pca[:2]):
        if v > 0.02:
            ax.text(x[i], v + 0.025, f"{v*100:.1f}%", ha="center", va="bottom", fontsize=6)

axes[0].set_ylabel("Variance fraction")
# no suptitle, no italic footnote — caption carries interpretation

out = pathlib.Path("experiments/dimensionality_3way.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=300, bbox_inches="tight")
print(f"saved to {out} {out.stat().st_size} bytes")
plt.close(fig)
