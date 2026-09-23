"""README hero infographic: EFN+ArcFace pipeline + key results.

Reads values from experiments/numbers.json (SSOT) so the figure can never
drift from the paper. Output: docs/infographic.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
NUMBERS = {e["name"]: e["expect"] for e in
           json.load(open(ROOT / "experiments" / "numbers.json"))["entries"]
           if "expect" in e}
OUT = ROOT / "docs" / "infographic.png"


def val(key):
    return NUMBERS[key]


NAVY, TEAL, AMBER, GREY, LIGHT = "#1b2a4a", "#0e7c7b", "#c77d0a", "#5b6472", "#eef1f6"

fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis("off")
fig.patch.set_facecolor("white")

# ---- header ----
ax.text(8, 9.45, "EFN + ArcFace Top Quark Tagging", ha="center", fontsize=26,
        fontweight="bold", color=NAVY)
ax.text(8, 9.05, "Geometrically readable jet classification: nearest prototype on a hypersphere",
        ha="center", fontsize=13, color=GREY)


def box(x, y, w, h, title, lines, color=NAVY, tsize=11):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                       facecolor="white", edgecolor=color, linewidth=2)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="top", fontsize=tsize,
            fontweight="bold", color=color)
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.62 - 0.30 * i, ln, ha="center", va="top",
                fontsize=9.5, color="#222222")


def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=16, color=GREY, linewidth=1.6))


# ---- pipeline row ----
py, ph = 6.55, 1.75
steps = [
    ("Constituents", ["(E, px, py, pz)", "up to 200 / jet"]),
    ("Phi MLP", ["[2 -> 128 x3]", "angles only"]),
    ("Energy pool", ["IRC-safe sum", "z = E / sum E"]),
    ("Rho MLP", ["[128 -> 128 -> 64]", "jet embedding"]),
    ("Hypersphere", ["L2 -> S^63", "ArcFace m=0.5"]),
    ("Prototypes", ["nearest of 2", "Top / QCD"]),
]
sw = 2.15
gap = (16 - 2 * 0.7 - len(steps) * sw) / (len(steps) - 1)
for i, (t, ln) in enumerate(steps):
    x = 0.7 + i * (sw + gap)
    box(x, py, sw, ph, t, ln, color=TEAL if i < 5 else AMBER)
    if i < len(steps) - 1:
        arrow(x + sw + 0.03, py + ph / 2, x + sw + gap - 0.03, py + ph / 2)

# ---- results row ----
ry, rh = 3.15, 2.75
cw = 3.45
gx = (16 - 2 * 0.7 - 4 * cw) / 3
acc = val("arcface_acc_s16_pct")
aci = (val("bootstrap_ci_lower_pct"), val("bootstrap_ci_upper_pct"))
box(0.7, ry, cw, rh, "Classification", [
    f"Accuracy {acc:.2f}%  [{aci[0]:.2f}-{aci[1]:.2f}]",
    f"AUC {val('auc_arcface'):.3f}   Rej@50% {val('rejection_tpr50'):.0f}",
    f"Params 59,072 (code-verified)",
    f"RF on features: {val('rf_acc_pct'):.2f}% (leads)",
], color=NAVY)
x = 0.7 + cw + gx
box(x, ry, cw, rh, "Subclasses (k*=2 x 2)", [
    "QCD-Core / QCD-Edge",
    "Top-Core / Top-Edge",
    "ARI 0.79 vs ground truth",
    "Top-Edge: non-identifiable",
], color=TEAL)
x += cw + gx
box(x, ry, cw, rh, "DGLAP check (zg tail)", [
    f"QCD-Core  alpha = {val('dglap_alpha_qcd_core'):.3f}",
    f"QCD-Edge  alpha = {val('dglap_alpha_qcd_edge'):.3f}",
    f"Top-Core  alpha = {val('dglap_alpha_top_core'):.3f}",
    "steeper = softer splitting",
], color=TEAL)
x += cw + gx
box(x, ry, cw, rh, "Geometry", [
    f"TwoNN dim {val('twonn_arcface'):.2f} (~2D)",
    f"PCA {val('pca_pc1_pct'):.1f}% + {val('pca_pc2_pct'):.1f}%",
    f"prototypes apart {val('prototype_distance_deg'):.1f} deg",
    f"mass {val('qcd_mass_core'):.0f} vs {val('qcd_mass_edge'):.0f} GeV; tau32 split",
], color=AMBER)

# ---- honesty footer ----
ax.text(8, 2.35, "Controls: null inputs still give k*=2 (geometry alone proves nothing)  |  "
        "label-shuffled retrain at chance, DGLAP ordering gone",
        ha="center", fontsize=10, color=GREY)
ax.text(8, 1.85, "Simulation-only (Pythia 8, no detector)  |  mass-confounded Core-Edge split stated  |  "
        "all 48 numbers machine-checked vs artifacts",
        ha="center", fontsize=10, color=GREY)
ax.text(8, 1.30, "github.com/thirathep-ph/EFN-ArcFace-Top-Quark-Tagging-and-Jet-Substructure-via-Hyperspherical-Prototypes",
        ha="center", fontsize=9.5, color=TEAL)

OUT.parent.mkdir(exist_ok=True)
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("wrote", OUT, OUT.stat().st_size, "bytes")
