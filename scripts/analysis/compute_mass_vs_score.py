"""
Diagnostic: Is the model just learning a mass cutoff?
=====================================================
Plots jet mass versus the model score P(Top|jet).

A model that relies solely on mass would produce a sharp vertical
decision boundary at :math:`m \\approx 150` GeV.  A model that learns
genuine substructure will exhibit a diffuse boundary where jets of
the *same* mass receive *different* classifications ― showing that
:math:`\\tau_{32}`, :math:`z_g`, and other substructure variables
contribute to the decision.

Output
------
``experiments/mass_vs_score_scatter.png``
    Two-panel figure: (left) scatter of 50k events;
    (right) 2-D histogram of all 404k events.
"""
import os, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from arcefn.utils.paths import EXPERIMENTS, DATA_DIR, CHECKPOINT
from arcefn.data.loader import load_awkward, JetTaggingDataset
from arcefn.models.efn_arcface import TopTaggingModel
from arcefn.utils.figure_style import apply_style, panel_label, add_grid, save_fig, COLORS, LEGEND_KWARGS
from arcefn.utils.physics import compute_features
from arcefn.utils.device import get_device

MAX_EVENTS = 50000
BATCH_SIZE = 2048


def get_embeddings_and_scores(model, events, labels, weights, device):
    """Run inference and return P(Top|jet) scores and true labels."""
    ds = JetTaggingDataset(events, labels, weights)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    all_top_sim = []
    all_labels = []
    with torch.no_grad():
        for x, y, w, m, _ in loader:
            x, m = x.to(device), m.to(device)
            logits, _, _ = model(x, mask=m)
            top_sim = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_top_sim.append(top_sim)
            all_labels.append(y.numpy())
    return np.concatenate(all_top_sim), np.concatenate(all_labels)


def main():
    print("=" * 60)
    print("Mass vs Score Scatter Plot")
    print("=" * 60)

    device, _ = get_device()

    model = TopTaggingModel(s=16.0, m=0.5, embedding_dim=64, particle_dim=128).to(device)
    sd = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(sd, strict=True)
    model.eval()

    test_h5 = os.path.join(str(DATA_DIR), "test.h5")

    print(f"\nStep 1: Loading test data ({MAX_EVENTS:,} events)...")
    t0 = time.time()
    events, labels, weights = load_awkward(test_h5, max_events=MAX_EVENTS, lazy=False)
    print(f"  Done in {time.time()-t0:.0f}s")

    print("\nStep 2: Computing embeddings and scores...")
    t0 = time.time()
    top_scores, _ = get_embeddings_and_scores(model, events, labels, weights, device)
    print(f"  Done in {time.time()-t0:.0f}s")

    print("\nStep 3: Computing jet mass via FastJet...")
    t0 = time.time()
    features = compute_features(events)
    jet_mass = features[:, 0]
    print(f"  Done in {time.time()-t0:.0f}s")

    print("\nStep 4: Plotting...")
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: scatter subsample
    ax = axes[0]
    n_plot = min(50000, len(labels))
    idx = np.random.RandomState(42).choice(len(labels), n_plot, replace=False)
    qcd = labels[idx] == 0
    top = labels[idx] == 1
    ax.scatter(jet_mass[idx][qcd], top_scores[idx][qcd],
               c=COLORS['qcd'], alpha=0.12, s=2, label='QCD', rasterized=True)
    ax.scatter(jet_mass[idx][top], top_scores[idx][top],
               c=COLORS['top'], alpha=0.12, s=2, label='Top', rasterized=True)
    ax.axhline(0.5, color='0.3', linestyle='--', lw=0.8, label='Decision boundary')
    ax.set_xlabel('Jet mass (GeV)')
    ax.set_ylabel('$P(\\mathrm{top} \\mid \\mathrm{jet})$')
    ax.legend(markerscale=5, **LEGEND_KWARGS)
    add_grid(ax)
    panel_label(ax, '(a)')

    # Right: 2D density all events (same x-range as left for consistency)
    ax = axes[1]
    from matplotlib.colors import LogNorm
    h = ax.hist2d(jet_mass, top_scores, bins=(80, 80), cmap='viridis',
                  range=[[0, 350], [0, 1]], norm=LogNorm(vmin=1))
    plt.colorbar(h[3], ax=ax, label='Count')
    ax.axhline(0.5, color='white', linestyle='--', lw=0.8, label='Decision boundary')
    ax.set_xlabel('Jet mass (GeV)')
    ax.set_ylabel('$P(\\mathrm{top} \\mid \\mathrm{jet})$')
    ax.legend(**LEGEND_KWARGS)
    add_grid(ax)
    panel_label(ax, '(b)')

    plt.tight_layout()
    save_path = os.path.join(str(EXPERIMENTS), 'mass_vs_score_scatter.png')
    save_fig(fig, 'mass_vs_score_scatter.png')
    print(f"\n  Saved: {save_path}")

    print(f"\n{'='*60}\nDIAGNOSTIC\n{'='*60}")
    for mass_cut in [100, 120, 140, 160, 180]:
        low = jet_mass < mass_cut
        high = jet_mass >= mass_cut
        qcd_like = (top_scores[low] < 0.5).mean() * 100
        top_like = (top_scores[high] >= 0.5).mean() * 100
        print(f"  Mass<{mass_cut:3d}GeV: {qcd_like:5.1f}% QCD | "
              f"Mass>={mass_cut:3d}GeV: {top_like:5.1f}% Top | "
              f"{high.mean()*100:.0f}% of data")
    print("\n[COMPLETE]")


if __name__ == '__main__':
    main()
