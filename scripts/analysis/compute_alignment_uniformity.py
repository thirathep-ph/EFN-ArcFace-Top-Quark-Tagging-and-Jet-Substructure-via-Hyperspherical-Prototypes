"""
Compare embedding geometry: ArcFace vs Linear head.
====================================================
Key insight: ArcFace forces all information into angular (directional)
degrees of freedom on S^63. Linear head can use norm as a proxy for
confidence, which trivializes clustering.

Output
------
* experiments/embedding_geometry_results.json
* experiments/embedding_geometry.png    — 2-panel figure
"""

import json, gc, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from arcefn.utils.paths import EXPERIMENTS, DATA_DIR
from arcefn.utils.model_loading import load_model_from_dir, read_model_config
from arcefn.data.loader import load_awkward, JetTaggingDataset
from arcefn.utils.device import get_device
from arcefn.utils.clustering import spectral_clustering_subclass
from arcefn.utils.figure_style import apply_style, panel_label, add_grid, save_fig, COLORS, LEGEND_KWARGS

TEST_H5 = str(DATA_DIR / 'test.h5')
N_EVENTS = 50000
BATCH_SIZE = 1024


def get_raw_embeddings(model, loader, device):
    """Get embeddings WITHOUT L2 normalization (plus logits for predicted labels)."""
    model.eval()
    all_emb, all_labels, all_logits = [], [], []
    with torch.no_grad():
        for x, y, w, m, _ in loader:
            x, m = x.to(device), m.to(device)
            out = model(x, mask=m)
            emb = out[-1]
            all_emb.append(emb.cpu().numpy())
            all_labels.append(y.numpy())
            all_logits.append(out[0].cpu().numpy())
    return np.vstack(all_emb), np.concatenate(all_labels), np.vstack(all_logits)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Embedding geometry comparison across model dirs (Auto detect ArcFace/CosLinear/Linear).")
    parser.add_argument("--model_dir", type=str, nargs='+',
                        default=[str(EXPERIMENTS / 'robustness_s16_m05'),
                                 str(EXPERIMENTS / 'baselines_coslinear_seed_42'),
                                 str(EXPERIMENTS / 'baselines_linear_seed_42')],
                        help="One or more experiment dirs (config + checkpoint.pt).")
    parser.add_argument("--out_tag", type=str, default="",
                        help="Suffix for output filenames, e.g. '_3way'.")
    parser.add_argument("--max_events", type=int, default=N_EVENTS,
                        help="Number of test events (default %(default)s).")
    args = parser.parse_args()

    device, _ = get_device()
    print(f"Device: {device}")
    print(f"Loading {args.max_events:,} events...")
    events, labels, weights = load_awkward(TEST_H5, max_events=args.max_events)
    ds = JetTaggingDataset(events, labels, weights)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model_dirs = [Path(d) for d in args.model_dir]
    models = []
    for d in model_dirs:
        info = read_model_config(d)
        arch = info['architecture']
        if arch == 'arcface':
            short = 'ArcFace (s=16, m=0.5)'
        elif arch == 'linear':
            short = 'Linear (cross-entropy)'
        else:
            short = 'CosLinear (m=0)'
        models.append((short, d))

    results = {}
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    colors = [COLORS['top'], COLORS['qcd'], '#7f7f7f']

    for idx, (name, ckpt_dir) in enumerate(models):
        print(f"\n--- {name} ---")
        t0 = time.time()
        model = load_model_from_dir(ckpt_dir, device)
        emb_raw, y, logits = get_raw_embeddings(model, loader, device)
        del model; gc.collect()
        from arcefn.utils.pred_labels import check_cm
        preds = np.argmax(logits, axis=1)
        check_cm(preds, y, name)
        y = preds  # cluster per PREDICTED class (what the model sees)
        norms = np.linalg.norm(emb_raw, axis=1)
        print(f"  Emb: {emb_raw.shape}, norms: min={norms.min():.4f}, median={np.median(norms):.4f}, max={norms.max():.4f}")

        # Pane 1: Norm histogram (raw embeddings for all models)
        ax = axes[0]
        ax.hist(norms, bins=80, color=colors[idx], alpha=0.6, density=True, label=name)

        # Pane 2: Subclass balance via spectral clustering
        # Normalize for spectral clustering (standard practice)
        emb_norm = emb_raw / (norms[:, None] + 1e-12)
        sub = np.full(len(y), -1, dtype=int)
        for cls, offset in [(0, 0), (1, 2)]:
            mask = y == cls
            if mask.sum() < 10: continue
            sl, k = spectral_clustering_subclass(emb_norm[mask], max_k=5)
            sub[mask] = sl + offset

        counts = {}
        for sc_id, sc_name in [(0, 'QCD_Core'), (1, 'QCD_Edge'), (2, 'Top_Core'), (3, 'Top_Edge')]:
            n = int(np.sum(sub == sc_id))
            counts[sc_name] = n
            results[f'{name.replace(chr(10), " ")}_{sc_name}_N'] = n

        results[f'{name.replace(chr(10), " ")}_norms_median'] = float(np.median(norms))
        results[f'{name.replace(chr(10), " ")}_norms_std'] = float(np.std(norms))

        ax = axes[1]
        sc_pos = np.arange(4)
        sc_vals = [counts['QCD_Core'], counts['QCD_Edge'],
                   counts['Top_Core'], counts['Top_Edge']]
        offset = idx * 0.3
        bars = ax.bar(sc_pos + offset, sc_vals, 0.3, color=colors[idx], alpha=0.8, label=name)
        # Annotate percentages
        total = sum(sc_vals)
        for bar, val in zip(bars, sc_vals):
            pct = val / total * 100
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 350,
                    f'{pct:.0f}%', ha='center', fontsize=7, rotation=90, va='bottom')

    # Format panel 1
    axes[0].set_xlabel('$\\|\\mathbf{e}\\|_2$')
    axes[0].set_ylabel('Density')
    axes[0].set_xlim(0, 25)
    axes[0].legend(fontsize=7, handlelength=1.2, **{k: v for k, v in LEGEND_KWARGS.items() if k not in ('fontsize',)})
    add_grid(axes[0])
    panel_label(axes[0], '(a)')

    # Format panel 2 — dynamic ylim to prevent overflow
    y_max = max((v for k, v in results.items() if k.endswith('_N')), default=22000)
    axes[1].set_ylim(0, y_max * 1.18)
    axes[1].set_ylabel('$N$ events (50k total)')
    axes[1].set_xticks(np.arange(4) + 0.15)
    axes[1].set_xticklabels(['QCD\nCore', 'QCD\nEdge', 'Top\nCore', 'Top\nEdge'], fontsize=7)
    axes[1].legend(fontsize=7, handlelength=1.2, **{k: v for k, v in LEGEND_KWARGS.items() if k not in ('fontsize',)})
    add_grid(axes[1], axis='y')
    panel_label(axes[1], '(b)')

    tag = args.out_tag or ""
    fig_name = f'embedding_geometry{tag}.png'
    json_name = f'embedding_geometry_results{tag}.json'
    plt.tight_layout(pad=0.6, w_pad=1.2)
    save_fig(fig, fig_name)
    print(f"\nSaved: {fig_name}")

    results_json = str(EXPERIMENTS / json_name)
    with open(results_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {results_json}")

    print("\n=== Key findings ===")
    for name_key in [nm for nm, _ in models]:
        mn = results.get(f'{name_key}_norms_median', '?')
        sd = results.get(f'{name_key}_norms_std', '?')
        print(f"  {name_key}: norm median={mn}, std={sd}")
        for sc in ['QCD_Core', 'QCD_Edge', 'Top_Core', 'Top_Edge']:
            n = results.get(f'{name_key}_{sc}_N', 0)
            print(f"    {sc}: {n}")


if __name__ == '__main__':
    main()
