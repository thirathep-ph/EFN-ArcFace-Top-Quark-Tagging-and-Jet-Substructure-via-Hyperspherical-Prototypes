"""README hero infographic v2: REAL experiment figures + tool stack.

No drawn boxes pretending to be results: every panel is a thumbnail of an
actual figure from docs/figures/ (same files linked in docs/RESULTS.md).
Tool chips name the software used at each pipeline stage (from
requirements.txt + code imports). Output: docs/infographic.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import image as mpimg

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "docs" / "figures"
OUT = ROOT / "docs" / "infographic.png"

NAVY, GREY = "#1b2a4a", "#5b6472"

# (file, caption) — all real figures, captions carry the numbers
PANELS = [
    ("architecture_schematic.png",
     "Model: EFN [2>128>128>128] + rho [128>128>64], 59,072 params, ArcFace m=0.5 s=16"),
    ("final_roc_curves.png",
     "Result: 91.84% [91.76, 91.92], AUC 0.974, rejection 124x (RF leads: 92.37%)"),
    ("dglap_fit.png",
     "Physics check: zg slopes -1.242 / -1.022 / -0.493 (Top-Edge excluded)"),
    ("lund_status_tp.png",
     "True-Top Lund plane: 3-prong structure"),
    ("lund_status_tn.png",
     "True-QCD Lund plane: single-prong radiation"),
    ("hypersphere_3d.png",
     "Geometry: embeddings collapse to ~2D disk (TwoNN 2.21, PCA 83.6+16.4)"),
]

TOOLS = [
    ("FastJet", "jet finding"),
    ("PyTorch", "models + training"),
    ("scikit-learn", "clustering + RF + metrics"),
    ("awkward+vector+h5py", "data IO"),
    ("matplotlib", "figures"),
    ("PySR", "symbolic distillation"),
]

fig = plt.figure(figsize=(20, 12))
fig.patch.set_facecolor("white")
gs = fig.add_gridspec(5, 3, height_ratios=[0.55, 2.2, 2.2, 0.65, 0.22],
                      hspace=0.45, wspace=0.25,
                      left=0.04, right=0.96, top=0.90, bottom=0.03)

fig.text(0.5, 0.96,
         "EFN-ArcFace Top Quark Tagging and Jet Substructure",
         ha="center", fontsize=24, fontweight="bold", color=NAVY)
fig.text(0.5, 0.925,
         "Nearest prototype on a hypersphere — every panel below is a real result figure",
         ha="center", fontsize=13, color=GREY)

for i, (fname, cap) in enumerate(PANELS):
    ax = fig.add_subplot(gs[1 + i // 3, i % 3])
    ax.imshow(mpimg.imread(FIG / fname))
    ax.axis("off")
    ax.set_title(cap, fontsize=9.5, color=NAVY, pad=4)

# tool stack row
tax = fig.add_subplot(gs[3, :])
tax.axis("off")
tax.set_xlim(0, 1)
tax.text(0.5, 0.82, "Built with", ha="center", fontsize=12,
         fontweight="bold", color=NAVY, transform=tax.transAxes)
for i, (tool, use) in enumerate(TOOLS):
    x = 0.03 + i * 0.16
    tax.text(x + 0.065, 0.45, tool, ha="center", fontsize=11,
             fontweight="bold", color="white",
             bbox=dict(boxstyle="round,pad=0.35", fc=NAVY, ec="none"),
             transform=tax.transAxes)
    tax.text(x + 0.065, 0.12, use, ha="center", fontsize=9,
             color=GREY, transform=tax.transAxes)
fax = fig.add_subplot(gs[4, :])
fax.axis("off")
fax.text(0.5, 0.5, "Simulation-only (Pythia 8)  |  null controls + single-seed shuffle disclosed  |  48 numbers machine-checked",
         ha="center", va="center", fontsize=9.5, color=GREY)

fig.savefig(OUT, dpi=110)
print("wrote", OUT, OUT.stat().st_size, "bytes")
