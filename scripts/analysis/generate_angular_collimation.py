"""
Generate Angular Collimation Figure: ΔR ≈ 2M/pT scaling per subclass
====================================================================
Uses the full 404k test set (N_EVENTS=404000) for physics features + clustering.
"""
import argparse
import gc
import json
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os

from arcefn.utils.paths import DATA_DIR
from arcefn.models.efn_arcface import TopTaggingModel
from arcefn.data.loader import load_awkward, JetTaggingDataset
from arcefn.utils.device import get_device
from arcefn.utils.physics import compute_features
from arcefn.utils.clustering import spectral_clustering_subclass
from arcefn.utils.figure_style import apply_style, panel_label, add_grid, save_fig, SUBCLASS_COLORS, LEGEND_KWARGS


def parse_args():
    parser = argparse.ArgumentParser(description="Generate angular collimation figure")
    parser.add_argument("--model_dir", type=str, default="experiments/robustness_s16_m05",
                        help="Directory containing model checkpoint")
    parser.add_argument("--embedding_dim", type=int, default=None,
                        help="Embedding dimension (overrides config.json if provided)")
    parser.add_argument("--max_events", type=int, default=404000,
                        help="Number of test events to use")
    parser.add_argument("--batch_size", type=int, default=1024,
                        help="Batch size for inference")
    return parser.parse_args()


TEST_H5 = str(DATA_DIR / 'test.h5')


def main():
    args = parse_args()
    
    model_dir = args.model_dir
    embedding_dim = args.embedding_dim
    max_events = args.max_events
    batch_size = args.batch_size
    
    device = get_device()[0]
    print(f"Device: {device}")

    # Load config from model_dir if exists
    config_path = os.path.join(model_dir, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        scale = config.get('scale', 16.0)
        margin = config.get('margin', 0.5)
        particle_dim = config.get('particle_dim', 128)
        if embedding_dim is None:
            embedding_dim = config.get('embedding_dim', 64)
    else:
        scale = 16.0
        margin = 0.5
        particle_dim = 128
        if embedding_dim is None:
            embedding_dim = 64

    print(f"Model config: scale={scale}, margin={margin}, embedding_dim={embedding_dim}, particle_dim={particle_dim}")

    model = TopTaggingModel(s=scale, m=0.0, embedding_dim=embedding_dim, particle_dim=particle_dim)
    ckpt = os.path.join(model_dir, 'checkpoint.pt')
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(sd, strict=True)
    model.to(device)
    model.arcface_head.set_margin(margin)
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    centers = sd['arcface_head.class_centers'].cpu().numpy()
    centers_norm = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-10)
    del sd; gc.collect()

    print(f"Loading {max_events} test events...")
    events, labels, weights = load_awkward(TEST_H5, max_events=max_events, lazy=False)
    labels = labels.numpy() if hasattr(labels, 'numpy') else np.array(labels)
    print(f"  Loaded {len(events)} events")

    print("Computing physics features (FastJet)...")
    t0 = time.time()
    phys = compute_features(events)
    print(f"  Done in {time.time()-t0:.0f}s, shape={phys.shape}")

    mass = phys[:, 0]
    theta_g = phys[:, 9]
    pt = np.sqrt(np.sum(events.px, axis=1)**2 + np.sum(events.py, axis=1)**2)
    pt = np.array(pt)
    two_m_pt = 2.0 * mass / np.clip(pt, 1.0, None)

    ds = JetTaggingDataset(events, labels, weights)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    all_emb, all_preds = [], []
    with torch.no_grad():
        for x, y, w, m, _ in loader:
            out = model(x.to(device), mask=m.to(device))
            logits, emb = out[0], out[2]
            all_emb.append(emb.cpu().numpy())
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
    embeddings = np.vstack(all_emb)
    predictions = np.concatenate(all_preds)

    emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)

    print("Spectral clustering per predicted class...")
    subclass_labels = np.full(len(predictions), -1, dtype=int)
    top_mask = predictions == 1
    active_top = np.array([], dtype=int)
    active_qcd = np.array([], dtype=int)
    if top_mask.sum() > 0:
        top_sc, k_top = spectral_clustering_subclass(emb_norm[top_mask], max_k=10)
        active_top, counts_top = np.unique(top_sc, return_counts=True)
        subclass_labels[top_mask] = top_sc + 1
        print(f"  Top subclasses: {dict(zip(active_top, counts_top))} k={k_top}")
    qcd_mask = predictions == 0
    if qcd_mask.sum() > 0:
        qcd_sc, k_qcd = spectral_clustering_subclass(emb_norm[qcd_mask], max_k=10)
        active_qcd, counts_qcd = np.unique(qcd_sc, return_counts=True)
        subclass_labels[qcd_mask] = -(qcd_sc + 1)
        print(f"  QCD subclasses: {dict(zip(active_qcd, counts_qcd))} k={k_qcd}")

    offset = 0
    sc_map = np.full(len(predictions), -1, dtype=int)
    if len(active_qcd) > 0:
        for i, sc in enumerate(active_qcd):
            sc_map[qcd_mask & (subclass_labels == -(sc + 1))] = offset + i
        offset += len(active_qcd)
    if len(active_top) > 0:
        for i, sc in enumerate(active_top):
            sc_map[top_mask & (subclass_labels == (sc + 1))] = offset + i
    sc_map_final = sc_map

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    qcd_sc_colors = [SUBCLASS_COLORS['QCD core'], SUBCLASS_COLORS['QCD edge'], '#66c2a5', '#2ca02c']
    top_sc_colors = [SUBCLASS_COLORS['Top core'], SUBCLASS_COLORS['Top edge'], '#9467bd', '#8c564b']

    max_plot = 25000
    rng = np.random.RandomState(42)
    for i, sc_idx in enumerate(active_qcd):
        mask = (sc_map_final == (i)) & (theta_g > 0.01) & (theta_g < 1.5) & (two_m_pt > 0.01) & (two_m_pt < 1.5)
        idx_plot = rng.choice(np.where(mask)[0], min(max_plot, mask.sum()), replace=False)
        label = f"Subclass {sc_idx} ($N={mask.sum():,}$)"
        axes[0].scatter(two_m_pt[idx_plot], theta_g[idx_plot], alpha=0.15, s=3,
                       color=qcd_sc_colors[i % len(qcd_sc_colors)], label=label)

    for i, sc_idx in enumerate(active_top):
        idx = len(active_qcd) + i
        mask = (sc_map_final == idx) & (theta_g > 0.01) & (theta_g < 1.5) & (two_m_pt > 0.01) & (two_m_pt < 1.5)
        idx_plot = rng.choice(np.where(mask)[0], min(max_plot, mask.sum()), replace=False)
        label = f"Subclass {sc_idx} ($N={mask.sum():,}$)"
        axes[1].scatter(two_m_pt[idx_plot], theta_g[idx_plot], alpha=0.15, s=3,
                       color=top_sc_colors[i % len(top_sc_colors)], label=label)

    for idx_ax, (ax, lbl) in enumerate(zip(axes, ["QCD", "Top"])):
        ax.plot([0, 1.5], [0, 1.5], 'k--', lw=0.8, label=r'$\Delta R = 2M/p_T$')
        ax.set_xlabel(r"$2M/p_T$")
        ax.set_ylabel(r"$\theta_g$")
        ax.set_xlim(0, 1.2); ax.set_ylim(0, 1.5)
        ax.legend(loc='lower right', **{**LEGEND_KWARGS, "fontsize": 7})
        add_grid(ax)
        panel_label(ax, f"({chr(ord('a') + idx_ax)})")

    plt.tight_layout()
    save_fig(fig, 'angular_collimation.png')

if __name__ == '__main__':
    import json
    parser = argparse.ArgumentParser(description="Generate angular collimation figure")
    parser.add_argument("--model_dir", type=str, default="experiments/robustness_s16_m05",
                        help="Directory containing model checkpoint")
    parser.add_argument("--embedding_dim", type=int, default=None,
                        help="Embedding dimension (overrides config.json if provided)")
    parser.add_argument("--max_events", type=int, default=404000,
                        help="Number of test events to use")
    parser.add_argument("--batch_size", type=int, default=1024,
                        help="Batch size for inference")
    args = parse_args()
    
    main()