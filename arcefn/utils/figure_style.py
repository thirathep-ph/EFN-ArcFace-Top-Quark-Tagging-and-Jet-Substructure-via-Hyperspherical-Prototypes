"""
Shared Figure Composition Standards for Publication Figures
===========================================================
All result figures should import and use these constants/functions
for visual consistency (font, color, legend, panel labels, grid).
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parents[2]
EXPERIMENTS = _REPO / "experiments"

# ── Typography ──────────────────────────────────────────────────────────────
FONT_FAMILY = "DejaVu Serif"
FONT_SIZE = 10
TICK_SIZE = 8
LEGEND_SIZE = 8
LABEL_SIZE = 10
TITLE_SIZE = 11

# ── Color Palette (muted, colorblind-safe) ─────────────────────────────────
COLORS = {
    "qcd":       "#377eb8",
    "top":       "#e41a1c",
    "qcd_edge":  "#9ecae1",
    "top_edge":  "#fdae6b",
    "green":     "#4daf4a",
    "purple":    "#984ea3",
    "orange":    "#ff7f00",
    "brown":     "#a65628",
    "gray":      "#999999",
}
CLASS_COLORS = {"QCD": COLORS["qcd"], "Top": COLORS["top"]}
SUBCLASS_COLORS = {
    "QCD core":  COLORS["qcd"],
    "QCD edge":  COLORS["qcd_edge"],
    "Top core":  COLORS["top"],
    "Top edge":  COLORS["top_edge"],
}

# ── Line / Marker Defaults ─────────────────────────────────────────────────
LINEWIDTH = 1.2
MARKERSIZE = 4
ALPHA = 0.7

# ── Legend ──────────────────────────────────────────────────────────────────
LEGEND_KWARGS = dict(
    frameon=True,
    fancybox=True,
    framealpha=0.92,
    edgecolor="0.45",
    facecolor="white",
    fontsize=LEGEND_SIZE,
    ncol=1,
)

# ── Grid ───────────────────────────────────────────────────────────────────
GRID_KWARGS = dict(alpha=0.15, linestyle="--", linewidth=0.4)

# ── Figure DPI ─────────────────────────────────────────────────────────────
DPI = 200


# ═══════════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════════

def apply_style():
    """Call once at top of script to set global rcParams."""
    mpl.rcParams.update({
        "font.family":      FONT_FAMILY,
        "font.size":        FONT_SIZE,
        "axes.labelsize":   LABEL_SIZE,
        "axes.titlesize":   TITLE_SIZE,
        "xtick.labelsize":  TICK_SIZE,
        "ytick.labelsize":  TICK_SIZE,
        "legend.fontsize":  LEGEND_SIZE,
        "figure.dpi":       DPI,
        "savefig.dpi":      DPI,
        "savefig.bbox":     "tight",
        "axes.linewidth":   0.6,
        "lines.linewidth":  LINEWIDTH,
        "lines.markersize": MARKERSIZE,
    })


def panel_label(ax, label, fontsize=TITLE_SIZE, loc="left", pad=6):
    """Add a bold panel label like $\mathbf{(a)}$ to an axes."""
    title = f"$\\mathbf{{{label}}}$"
    if loc == "left":
        ax.set_title(title, fontsize=fontsize, loc="left", fontweight="bold", pad=pad)
    else:
        ax.set_title(title, fontsize=fontsize, loc=loc, pad=pad)


def style_legend(ax, **kwargs):
    """Apply consistent legend styling. Pass extra kwargs to override defaults."""
    kw = {**LEGEND_KWARGS, **kwargs}
    leg = ax.get_legend()
    if leg is not None:
        leg.set_frame(**{k: v for k, v in kw.items()
                         if k in ("frameon", "fancybox", "framealpha", "edgecolor", "facecolor")})
        if "fontsize" in kw:
            for t in leg.get_texts():
                t.set_fontsize(kw["fontsize"])
        if "ncol" in kw:
            leg._ncol = kw["ncol"]


def add_grid(ax, **kwargs):
    """Add grid with consistent styling."""
    kw = {**GRID_KWARGS, **kwargs}
    ax.grid(True, **kw)
    ax.set_axisbelow(True)


def save_fig(fig, name, **kwargs):
    """Save figure to experiments/. Gallery figures in docs/figures/ are frozen and never overwritten by code.
    Uses ARCEFN_MODEL_DIR env var for model-specific subdirectory, falls back to EXPERIMENTS.
    """
    import os
    model_dir_env = os.environ.get('ARCEFN_MODEL_DIR', '').strip()
    if model_dir_env:
        exp_path = Path(model_dir_env) / name
    else:
        exp_path = EXPERIMENTS / name
    exp_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(exp_path), **{"dpi": DPI, "bbox_inches": "tight", **kwargs})
    plt.close(fig)
    print(f"  Saved: {name}")
