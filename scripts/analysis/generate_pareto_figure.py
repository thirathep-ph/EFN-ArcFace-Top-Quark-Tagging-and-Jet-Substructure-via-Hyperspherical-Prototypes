"""Generate SciPost-style Pareto front figure for symbolic regression results.

SciPost style: minimal, grayscale-friendly, LaTeX math labels, no gimmicky
shading or AI-style annotations. Standard scientific scatter with key
equations annotated in a legend-like inset.
"""
from __future__ import annotations

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from arcefn.utils.paths import EXPERIMENTS
from arcefn.utils.figure_style import add_grid, save_fig, LEGEND_KWARGS

FIT_PATH = EXPERIMENTS / "physics_sr" / "fit_full.json"
FIGURES = EXPERIMENTS.parent / "paper" / "figures"

SELECTED_C = 6


def has_nested(eq: str) -> bool:
    import re
    for fn in ("tanh", "exp", "log", "sqrt"):
        if re.search(rf"\b{fn}\s*\([^)]*\b{fn}\s*\(", eq):
            return True
    return False


def count_features(eq: str) -> int:
    names = ["Mass_over_pT", "mSD_over_pT", "Mult", "nSD",
             "sqrt_d12", "sqrt_d23", "Tau21", "Tau32", "zg", "theta_g"]
    return max(1, sum(1 for f in names if f in eq))


def short_label(eq: str, max_len: int = 55) -> str:
    eq = eq.replace("Mass_over_pT", "m/p_T").replace("mSD_over_pT", "m_{SD}/p_T")
    eq = eq.replace("Tau32", "\\tau_{32}").replace("Tau21", "\\tau_{21}")
    eq = eq.replace("theta_g", "\\theta_g").replace("zg", "z_g")
    if len(eq) > max_len:
        eq = eq[:max_len - 3] + "..."
    return eq


def main():
    data = json.loads(FIT_PATH.read_text(encoding="utf-8"))
    eqs = data["equations"]

    C = np.array([e["complexity"] for e in eqs])
    L = np.array([e["loss"] for e in eqs])
    nested = np.array([has_nested(e["equation"]) for e in eqs])

    # --- Publication styling (SciPost / matplotlib best practice) ---
    # OO interface, constrained_layout, serif CM, explicit grid/ticks, colorblind palette
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "mathtext.fontset": "cm",
        "axes.titlesize": 11, "axes.labelsize": 11,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "axes.linewidth": 1.0, "xtick.major.width": 0.9,
        "ytick.major.width": 0.9, "xtick.minor.width": 0.6, "ytick.minor.width": 0.6,
        "xtick.major.size": 3.5, "ytick.major.size": 3.5,
        "xtick.minor.size": 2.0, "ytick.minor.size": 2.0,
    })

    fig, ax = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)

    # Pareto front as dashed baseline (academic standard: front is envelope, not data line)
    ax.plot(C, L, color="#6e6e6e", alpha=0.9, linewidth=1.2, linestyle=(0, (4, 3)),
            zorder=1)

    # Plateu shading — where adding complexity barely helps (C>14, loss <0.09)
    ax.axhspan(0.05, 0.095, xmin=14/24, xmax=1.0, facecolor="#f0f0f0", edgecolor="none", zorder=0)
    ax.text(21.5, 0.073, "plateau", fontsize=7, color="#6e6e6e", ha="right", va="center", style="italic")

    # Colorblind Wong palette: blue #0072B2, vermillion #D55E00, gold #CCB000
    clean_mask = ~nested
    dirty_mask = nested

    # Smaller icons — was slightly oversized, now compact for publication
    ax.scatter(C[clean_mask], L[clean_mask], marker='o', s=52,
               facecolors='none', edgecolors='#0072B2', linewidths=1.4,
               zorder=3, label='No nested functions')
    ax.scatter(C[dirty_mask], L[dirty_mask], marker='x', s=54,
               color='#D55E00', linewidths=1.5,
               zorder=3, label='Contains nested')

    sel_idx = np.where(C == SELECTED_C)[0][0]
    ax.scatter([C[sel_idx]], [L[sel_idx]], marker='D', s=78,
               facecolors='#CCB000', edgecolors='black', linewidths=1.0,
               zorder=5, label=f'Selected ($C={SELECTED_C}$)')
    ax.scatter([C[sel_idx]], [L[sel_idx]], marker='D', s=105,
               facecolors='none', edgecolors='#CCB000', linewidths=2.0, alpha=0.32, zorder=4)

    sel_y = float(L[np.where(C == SELECTED_C)[0][0]])
    ax.plot([SELECTED_C, SELECTED_C], [0.0, sel_y], color="#CCB000", alpha=0.60, linewidth=1.1, linestyle=(0, (3, 2.5)), zorder=1)
    ax.text(SELECTED_C, sel_y + 0.038, r"$C=6$", fontsize=7.5, color="#7a6200", ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#CCB000", alpha=0.98, linewidth=0.65),
            clip_on=False, zorder=6)

    # Annotations — offset first label off the x=2 grid line so leader does not coincide with grid/vertical
    import matplotlib.patheffects as pe
    annotations = {
        2:  {"label": r"$\tanh(m/p_T)$", "dx": 0.90, "dy": 0.022, "ha": "left", "va": "bottom"},
        6:  {"label": r"$\tanh(m/p_T)-0.35\,\tau_{32}$", "dx": 1.35, "dy": 0.032, "ha": "left", "va": "bottom"},
    }
    for c, ann in annotations.items():
        idx = np.where(C == c)[0]
        if len(idx) == 0:
            continue
        i = idx[0]
        x, y = C[i], L[i]
        ax.annotate(
            ann["label"],
            xy=(x, y), xytext=(x + ann["dx"], y + ann["dy"]),
            fontsize=9.5, ha=ann["ha"], va=ann["va"], color="#1a1a1a",
            arrowprops=dict(arrowstyle="-", color="#333333", lw=0.9, shrinkB=3),
            path_effects=[pe.withStroke(linewidth=3.5, foreground="white")],
            zorder=6,
        )

    ax.set_xlabel("Expression complexity $C$", fontsize=11)
    ax.set_ylabel("MSE loss (z-scored logit)", fontsize=11)
    ax.set_xlim(0, 24)
    ax.set_ylim(0.0, 0.315)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(2))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.025))
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.grid(True, which="major", linestyle="-", linewidth=0.7, alpha=0.32, color="#222222")
    ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.18, color="#555555")
    ax.set_axisbelow(True)

    # Legend outside data, thin border — SciPost single-col friendly
    leg = ax.legend(loc='upper right', **{**LEGEND_KWARGS, "fontsize": 8.5, "framealpha": 1.0, "edgecolor": "#333333"},
                    handlelength=1.7, handletextpad=0.5, borderpad=0.45, labelspacing=0.35)
    leg.get_frame().set_linewidth(0.8)

    save_fig(fig, 'pareto_front_sr.png', dpi=300)
    print(f"Saved: pareto_front_sr.png")


if __name__ == "__main__":
    main()
