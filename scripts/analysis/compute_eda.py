"""Exploratory Data Analysis for Top Tagging Reference Dataset.

Generates 4 figures + 1 statistics table for the paper appendix.
All kinematics computed from full training set (1.2M events).
Substructure features from 100K training subset via FastJet.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

sys.path.append(str(Path(__file__).resolve().parent.parent))  # noqa: E402
from arcefn.data.loader import load_awkward, get_h5_len
from arcefn.utils.physics import compute_features, print_feature_stats
from arcefn.utils.device import get_device
from arcefn.utils.paths import DATA_DIR, EXPERIMENTS
from arcefn.utils.figure_style import apply_style, panel_label, add_grid, save_fig, COLORS, LEGEND_KWARGS
OUT = EXPERIMENTS / 'eda'
OUT.mkdir(parents=True, exist_ok=True)

TRAIN_H5 = str(DATA_DIR / 'train.h5')
VAL_H5 = str(DATA_DIR / 'val.h5')
TEST_H5 = str(DATA_DIR / 'test.h5')
N_SUBSET = 100_000


# ── 1. Dataset statistics ───────────────────────────────────────────
def compute_split_stats(path: str, label: str) -> dict:
    """Compute basic statistics for one dataset split (no FastJet)."""
    events, labels, _ = load_awkward(path, lazy=True, trim_padding=True)
    es, pxs, pys, pzs = events.E, events.px, events.py, events.pz
    mask = es > 0
    n_per = mask.sum(axis=1).astype(np.float32)

    total_E = es.sum(axis=1)
    total_px = pxs.sum(axis=1)
    total_py = pys.sum(axis=1)
    total_pz = pzs.sum(axis=1)

    jet_pt = np.sqrt(total_px**2 + total_py**2)
    jet_eta = 0.5 * np.log(np.clip((total_E + total_pz) / (total_E - total_pz + 1e-10), 1e-10, 1e10))
    jet_mass = np.sqrt(np.clip(total_E**2 - total_px**2 - total_py**2 - total_pz**2, 0, None))

    stats = {
        'split': label,
        'n_events': len(labels),
        'frac_top': float(labels.mean()),
        'mean_constituents': float(n_per.mean()),
        'mean_pt': float(jet_pt.mean()),
        'mean_eta': float(np.abs(jet_eta).mean()),
        'mean_mass': float(jet_mass.mean()),
    }
    print(f"  {label}: {stats['n_events']:,} events, "
          f"{stats['frac_top']*100:.1f}% top, "
          f"{stats['mean_constituents']:.1f} mean constituents")
    return stats, labels, jet_pt, jet_eta, jet_mass, n_per


def collect_stats() -> tuple[list[dict], dict]:
    stats_list = []
    full_labels, full_pt, full_eta, full_mass, full_mult = [], [], [], [], []
    for path, label in [(TRAIN_H5, 'train'), (VAL_H5, 'val'), (TEST_H5, 'test')]:
        s, labels, pt, eta, mass, mult = compute_split_stats(path, label)
        stats_list.append(s)
        if label == 'train':
            full_labels.append(labels)
            full_pt.append(pt)
            full_eta.append(eta)
            full_mass.append(mass)
            full_mult.append(mult)
    full = {
        'labels': np.concatenate(full_labels),
        'pt': np.concatenate(full_pt),
        'eta': np.concatenate(full_eta),
        'mass': np.concatenate(full_mass),
        'mult': np.concatenate(full_mult),
    }
    return stats_list, full


# ── 2. Substructure features (100K FastJet) ─────────────────────────
def compute_substructure() -> dict:
    """Compute all 10 FastJet features on training subset."""
    path = str(DATA_DIR / 'train.h5')
    print(f"Loading {N_SUBSET:,} events for FastJet features...")
    events, labels, _ = load_awkward(path, max_events=N_SUBSET, lazy=False)
    print("Computing substructure features...")
    feats = compute_features(events)
    print_feature_stats(feats)
    return {
        'features': feats,
        'labels': labels,
        'names': ["Mass", "mSD", "Mult", "nSD", r"$\sqrt{d_{12}}$", r"$\sqrt{d_{23}}$",
                   r"$\tau_{21}$", r"$\tau_{32}$", r"$z_g$", r"$\theta_g$"],
        'tex_names': ["Mass", r"$m_{SD}$", r"Mult", r"$n_{SD}$",
                      r"$\sqrt{d_{12}}$", r"$\sqrt{d_{23}}$",
                      r"$\tau_{21}$", r"$\tau_{32}$", r"$z_g$", r"$\theta_g$"],
    }


# ── 3. Plotting ─────────────────────────────────────────────────────
def plot_fig1_class_balance(stats_list: list[dict], outdir: Path):
    """Fig A1: Dataset overview — split sizes + class balance."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    splits = [s['split'] for s in stats_list]
    n_events = [s['n_events'] / 1e6 for s in stats_list]

    colors = [COLORS["gray"], COLORS["gray"], COLORS["gray"]]
    bars = axes[0].bar(splits, n_events, color=colors, edgecolor='white', width=0.5)
    axes[0].set_ylabel("Events (millions)")
    axes[0].set_title("Dataset Splits")
    for bar, n in zip(bars, n_events):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f"{n:.2f}M", ha='center', va='bottom', fontsize=10)
    axes[0].set_ylim(0, max(n_events) * 1.15)

    # Class balance bar chart
    top_frac = [s['frac_top'] * 100 for s in stats_list]
    qcd_frac = [100 - tf for tf in top_frac]
    x = np.arange(len(splits))
    width = 0.35
    axes[1].bar(x - width / 2, qcd_frac, width, label='QCD', color=COLORS["qcd"], edgecolor='white')
    axes[1].bar(x + width / 2, top_frac, width, label='Top', color=COLORS["top"], edgecolor='white')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(splits)
    axes[1].set_ylabel("Percentage (%)")
    axes[1].set_title("Class Balance")
    axes[1].legend()
    axes[1].set_ylim(0, 100)

    plt.tight_layout()
    save_fig(fig, 'eda_figA1_dataset_overview.png')


def plot_fig2_kinematics(full: dict, outdir: Path):
    """Fig A2: Kinematics — pT, eta, mass, multiplicity (2x2)."""
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    labels = full['labels']
    top_mask = labels == 1
    qcd_mask = labels == 0

    def plot_kde(ax, data_top, data_qcd, xlabel, title, bins=80, xlim=None):
        for data, label, color in [(data_qcd, 'QCD', COLORS["qcd"]), (data_top, 'Top', COLORS["top"])]:
            ax.hist(data, bins=bins, density=True, alpha=0.5, color=color,
                    label=label, histtype='stepfilled')
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.set_title(title)
        ax.legend(fontsize=9)
        if xlim:
            ax.set_xlim(xlim)

    plot_kde(axes[0, 0], full['pt'][top_mask], full['pt'][qcd_mask],
             r"$p_T$ (GeV)", r"Jet $p_T$ Distribution", xlim=(500, 700))
    plot_kde(axes[0, 1], full['eta'][top_mask], full['eta'][qcd_mask],
             r"$\eta$", r"Pseudorapidity Distribution")
    plot_kde(axes[1, 0], full['mass'][top_mask], full['mass'][qcd_mask],
             r"Mass (GeV)", r"Jet Mass Distribution", xlim=(0, 400))
    plot_kde(axes[1, 1], full['mult'][top_mask], full['mult'][qcd_mask],
             "Number of constituents", r"Constituent Multiplicity", xlim=(0, 200))

    plt.tight_layout()
    save_fig(fig, 'eda_figA2_kinematics.png')


def plot_fig3_substructure(sub: dict, outdir: Path):
    """Fig A3: All 10 substructure features (2x5 grid)."""
    apply_style()
    feats, labels = sub['features'], sub['labels']
    names_tex = sub['tex_names']
    top_mask = labels == 1
    qcd_mask = labels == 0

    n_feats = feats.shape[1]
    n_cols = 5
    n_rows = int(np.ceil(n_feats / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 7))
    axes = axes.flatten()

    for i in range(n_feats):
        ax = axes[i]
        d = feats[:, i]
        valid = ~np.isnan(d) & ~np.isinf(d) & (d > 0)
        d_qcd = d[valid & qcd_mask]
        d_top = d[valid & top_mask]

        lo, hi = np.percentile(d_qcd, [0.5, 99.5])
        lo2, hi2 = np.percentile(d_top, [0.5, 99.5])
        lo = min(lo, lo2)
        hi = max(hi, hi2)
        bins = np.linspace(lo, hi, 60)

        ax.hist(d_qcd, bins=bins, density=True, alpha=0.5, color=COLORS["qcd"],
                label='QCD' if i == 0 else '', histtype='stepfilled')
        ax.hist(d_top, bins=bins, density=True, alpha=0.5, color=COLORS["top"],
                label='Top' if i == 0 else '', histtype='stepfilled')
        ax.set_xlabel(names_tex[i], fontsize=11)
        ax.set_ylabel("Density", fontsize=10)
        ax.ticklabel_format(style='sci', scilimits=(-3, 3), axis='y')
        ax.tick_params(labelsize=9)

    for j in range(n_feats, len(axes)):
        axes[j].set_visible(False)

    fig.legend(['QCD', 'Top'], loc='upper right', fontsize=12,
               bbox_to_anchor=(0.98, 0.98))
    fig.suptitle("Substructure Feature Distributions (100K training events)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fpath = outdir / 'eda_figA3_substructure.png'
    fig.savefig(fpath, bbox_inches='tight', dpi=150)
    print(f"  Saved {fpath}")
    plt.close(fig)


def plot_fig4_correlation(sub: dict, outdir: Path):
    """Fig A4: Feature correlation heatmap."""
    feats = sub['features']
    names_short = sub['names']

    corr = np.corrcoef(feats.T)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = plt.cm.RdBu_r
    im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1)

    ax.set_xticks(range(len(names_short)))
    ax.set_yticks(range(len(names_short)))
    ax.set_xticklabels(names_short, rotation=45, ha='right')
    ax.set_yticklabels(names_short)

    for i in range(len(names_short)):
        for j in range(len(names_short)):
            if not mask[i, j]:
                val = corr[i, j]
                color = 'white' if abs(val) > 0.5 else 'black'
                ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                        fontsize=6, color=color)

    fig.colorbar(im, ax=ax, shrink=0.8, label='Pearson $r$')
    plt.tight_layout()
    save_fig(fig, 'eda_figA4_correlation.png')


# ── Main ─────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("EDA: Top Tagging Reference Dataset")
    print("=" * 60)

    print("\n[1/4] Dataset statistics (full splits)...")
    stats_list, full = collect_stats()

    print("\n[2/4] Substructure features (100K FastJet)...")
    sub = compute_substructure()

    print("\n[3/4] Generating figures...")
    plot_fig1_class_balance(stats_list, OUT)
    plot_fig2_kinematics(full, OUT)
    plot_fig3_substructure(sub, OUT)
    plot_fig4_correlation(sub, OUT)

    print("\n[4/4] Saving statistics...")
    result = {
        'dataset_stats': stats_list,
        'correlation_matrix': np.corrcoef(sub['features'].T).tolist(),
        'feature_names': sub['names'],
    }
    jpath = OUT / 'eda_stats.json'
    # Convert numpy types for JSON serialization
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            return super().default(obj)
    with open(jpath, 'w') as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)
    print(f"  Saved {jpath}")

    # Print summary table
    print("\n" + "=" * 60)
    print("DATASET STATISTICS SUMMARY")
    print("=" * 60)
    print(f"{'Split':<8} {'N Events':<12} {'%Top':<8} {'<pT>':<10} {'<|eta|>':<10} {'<Mass>':<10} {'<N_cons>':<10}")
    print("-" * 68)
    for s in stats_list:
        print(f"{s['split']:<8} {s['n_events']:<12,} {s['frac_top']*100:<8.1f} "
              f"{s['mean_pt']:<10.1f} {s['mean_eta']:<10.3f} "
              f"{s['mean_mass']:<10.1f} {s['mean_constituents']:<10.1f}")

    print(f"\nAll EDA outputs in {OUT}")


if __name__ == '__main__':
    main()
