"""
Eigengap Spectrum Plot
======================
Plots eigenvalues of the normalized Laplacian for QCD and Top embeddings,
highlighting the largest gap which determines optimal k (number of subclasses).

Saves figure to ``EXPERIMENTS / 'eigengap_spectrum.png'``.

Examples
--------
Run from the project root::

    python scripts/analysis/compute_eigengap.py
"""
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from arcefn.utils.paths import DATA_DIR, EXPERIMENTS, MODEL_DIR, CHECKPOINT
from arcefn.utils.device import get_device
from arcefn.data.loader import load_awkward, JetTaggingDataset
from arcefn.models.efn_arcface import TopTaggingModel
from arcefn.utils.clustering import estimate_clusters_eigengap
from arcefn.utils.figure_style import apply_style, panel_label, add_grid, save_fig, COLORS, LEGEND_KWARGS

MAX_EVENTS = 80000
BATCH_SIZE = 2048


def main():
    """Run eigengap analysis: load model, compute embeddings, plot spectrum."""
    print("=" * 60)
    print("Eigengap Spectrum Plot")
    print("=" * 60)

    device, _ = get_device()

    model = TopTaggingModel(s=16.0, m=0.5, embedding_dim=64, particle_dim=128).to(device)
    sd = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(sd, strict=True)
    model.eval()

    test_h5 = DATA_DIR / 'test.h5'
    print(f"\nLoading test data ({MAX_EVENTS:,} events)...")
    t0 = time.time()
    events, labels, weights = load_awkward(test_h5, max_events=MAX_EVENTS, lazy=False)
    print(f"  Done in {time.time()-t0:.0f}s")

    ds = JetTaggingDataset(events, labels, weights)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print("Computing embeddings...")
    t0 = time.time()
    all_embeddings = []
    all_preds = []
    with torch.no_grad():
        for x, y, w, m, _ in loader:
            x, m = x.to(device), m.to(device)
            logits, _, jet_emb = model(x, mask=m)
            emb = F.normalize(jet_emb, p=2, dim=1)
            all_embeddings.append(emb.cpu().numpy())
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.append(preds)
    embeddings = np.concatenate(all_embeddings)
    preds = np.concatenate(all_preds)
    print(f"  Done in {time.time()-t0:.0f}s")

    print("Computing eigengap spectra...")
    results = {}
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, (class_name, class_val) in enumerate([("QCD", 0), ("Top", 1)]):
        mask = preds == class_val
        X = embeddings[mask]
        print(f"  {class_name}: {len(X)} events")

        k_opt, evals = estimate_clusters_eigengap(X, max_k=10)

        ax = axes[idx]
        n_show = min(8, len(evals))
        ks = np.arange(1, n_show + 1)
        color = COLORS['qcd'] if class_name == "QCD" else COLORS['top']
        ax.plot(ks, evals[:n_show], 'o-', color=color, markersize=5, linewidth=1.2, label='Eigenvalues')
        ax.axvline(k_opt, color='0.3', linestyle='--', linewidth=0.8,
                   label=f'Optimal $k$ = {k_opt}')

        # Inset: eigengap bar chart
        gaps = np.diff(evals[:n_show])
        gap_ks = np.arange(2, n_show + 1)
        ax2 = ax.inset_axes([0.35, 0.35, 0.6, 0.55])
        bar_colors = [COLORS['top'] if k == k_opt else '0.65' for k in gap_ks]
        ax2.bar(gap_ks, gaps, width=0.6, color=bar_colors, alpha=0.8, edgecolor='white', linewidth=0.5)
        ax2.axvline(k_opt, color='0.3', linestyle='--', linewidth=0.8)
        ax2.set_xlabel('$k$', fontsize=9)
        ax2.set_ylabel('Gap', fontsize=9)
        ax2.tick_params(labelsize=8)
        ax2.set_title('Eigengaps', fontsize=9, pad=2)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        ax.set_xlabel('Eigenvalue index $k$')
        ax.set_ylabel(r'$\lambda_k$')
        ax.set_xticks(ks)
        ax.legend(loc='lower right', **LEGEND_KWARGS)
        add_grid(ax)
        ax.set_xlim(0.5, n_show + 0.5)
        panel_label(ax, f"({chr(ord('a') + idx)})")

        results[class_name] = {
            'optimal_k': int(k_opt),
            'n_events': len(X),
            'eigenvalues': [float(v) for v in evals],
        }

    plt.tight_layout()
    save_path = EXPERIMENTS / 'eigengap_spectrum.png'
    save_fig(fig, 'eigengap_spectrum.png')
    print(f"\n  Saved: {save_path}")

    print(f"\nResults: QCD optimal_k={results['QCD']['optimal_k']}, Top optimal_k={results['Top']['optimal_k']}")
    print("[COMPLETE]")


if __name__ == '__main__':
    main()
