"""
Paper-Quality Figures for EFN + ArcFace
=========================================
Generates the main paper figures from the trained model:

* 2D UMAP coloured by prediction status (TP / TN / FP / FN)
* 2D UMAP coloured by spectral-clustering subclasses
* Subclass physics-distribution KDE plots (mass, tau32, sqrt(d12))
* Subclass composition heatmap (purity per cluster)
* Lund-plane plots for TP, TN, FP, FN

Uses stratified sampling: 50 k events for UMAP, ~15 k for FastJet.

Output
------
All figures are saved under ``MODEL_DIR`` (default: ``experiments/robustness_s16_m05/``):

* ``umap_prediction_status.png``
* ``umap_subclasses.png``
* ``subclass_physics_distributions.png``
* ``subclass_composition_heatmap.png``
* ``lund_planes/status_tp.png``, ``…tn.png``, ``…fp.png``, ``…fn.png``
"""

import argparse
import os
import time
import gc
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from arcefn.utils.paths import DATA_DIR
from arcefn.models.efn_arcface import TopTaggingModel
from arcefn.data.loader import load_awkward, JetTaggingDataset
from arcefn.utils.device import get_device
from arcefn.utils.figure_style import (
    apply_style, panel_label, add_grid, save_fig,
    COLORS, SUBCLASS_COLORS, LEGEND_KWARGS, PAPER_FIGURES,
)
apply_style()
from arcefn.utils.physics import compute_features
from arcefn.utils.clustering import spectral_clustering_subclass
from arcefn.utils.plotting import plot_lund_plane

TEST_H5 = str(DATA_DIR / 'test.h5')
MAX_EVENTS = 100000
UMAP_SAMPLE = 50000
PHYS_SAMPLE = 15000
BATCH_SIZE = 1024


def parse_args():
    parser = argparse.ArgumentParser(description="Generate paper figures from trained model")
    parser.add_argument("--model_dir", type=str, default="experiments/robustness_s16_m05",
                        help="Directory containing model checkpoint and config")
    parser.add_argument("--embedding_dim", type=int, default=None,
                        help="Embedding dimension (overrides config.json if provided)")
    parser.add_argument("--max_events", type=int, default=MAX_EVENTS,
                        help="Number of test events to use")
    parser.add_argument("--umap_sample", type=int, default=UMAP_SAMPLE,
                        help="Number of events for UMAP")
    parser.add_argument("--phys_sample", type=int, default=PHYS_SAMPLE,
                        help="Number of events for physics/Lund plots")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Batch size for inference")
    parser.add_argument("--partition", type=str, choices=["predicted", "true"], default="predicted",
                        help="partition for subclass discovery: predicted (model view) or true (GT check)")
    return parser.parse_args()


def compute_embeddings_and_labels(model, events, labels, weights, device):
    """
    Run model inference and return embeddings, predictions, and labels.

    Parameters
    ----------
    model : torch.nn.Module
        Trained EFN + ArcFace model.
    events : awkward array
        Jet constituent events.
    labels : ndarray
        Ground-truth class labels.
    weights : ndarray
        Event weights.
    device : torch.device
        Compute device.

    Returns
    -------
    embeddings : ndarray, shape (N, D)
    preds : ndarray, shape (N,)
    labels : ndarray, shape (N,)
    """
    ds = JetTaggingDataset(events, labels, weights)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_emb, all_preds, all_labels = [], [], []

    with torch.no_grad():
        for x, y, w, m, _ in loader:
            out = model(x.to(device), mask=m.to(device))
            if isinstance(out, (tuple, list)):
                logits = out[0]
                emb = out[-1]
            else:
                logits, emb = out, out
            all_emb.append(emb.cpu().numpy())
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_labels.append(y.numpy())

    return np.vstack(all_emb), np.concatenate(all_preds), np.concatenate(all_labels)


def stratified_sample(labels, n_per_class, rng=None):
    """
    Draw a stratified sample of indices with equal class counts.

    Parameters
    ----------
    labels : ndarray
        Ground-truth or predicted labels.
    n_per_class : int
        Number of samples per class.
    rng : numpy.random.RandomState or None
        Random state for reproducibility.

    Returns
    -------
    indices : ndarray
        Sampled indices (shuffled).
    """
    if rng is None:
        rng = np.random.RandomState(42)
    idx = []
    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        n = min(n_per_class, len(cls_idx))
        idx.append(rng.choice(cls_idx, n, replace=False))
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return idx


def main():
    """
    Generate all main paper figures.

    Loads the trained model and all test events, computes embeddings,
    performs stratified 50 k sampling for UMAP + spectral clustering,
    then ~15 k sampling for FastJet physics and Lund-plane visualisation.
    """
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Setup
    # ------------------------------------------------------------------
    device, _ = get_device()
    print(f"Device: {device}\n")

    # Load model configuration
    model_dir = args.model_dir
    os.makedirs(model_dir, exist_ok=True)
    lund_dir = os.path.join(model_dir, 'lund_planes')
    os.makedirs(lund_dir, exist_ok=True)

    # Load config from model_dir if exists
    config_path = os.path.join(model_dir, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        scale = config.get('scale', 16.0)
        margin = config.get('margin', 0.5)
        particle_dim = config.get('particle_dim', 128)
        if args.embedding_dim is None:
            embedding_dim = config.get('embedding_dim', 64)
        else:
            embedding_dim = args.embedding_dim
    else:
        scale = 16.0
        margin = 0.5
        particle_dim = 128
        embedding_dim = args.embedding_dim if args.embedding_dim is not None else 64

    print(f"Model config: scale={scale}, margin={margin}, embedding_dim={embedding_dim}, particle_dim={particle_dim}")

    # ------------------------------------------------------------------
    # 2. Load model
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Loading model")
    print("=" * 60)

    checkpoint_path = os.path.join(model_dir, 'checkpoint.pt')
    model = TopTaggingModel(s=scale, m=0.0, embedding_dim=embedding_dim, particle_dim=particle_dim)
    sd = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(sd, strict=True)
    model.to(device)
    model.eval()
    model.arcface_head.set_margin(margin)

    # Extract class centres
    if 'arcface_head.class_centers' in sd:
        class_centers = sd['arcface_head.class_centers'].cpu().numpy()
    else:
        raise KeyError("'arcface_head.class_centers' not found in state_dict.")
    centers_norm = class_centers / (np.linalg.norm(class_centers, axis=1, keepdims=True) + 1e-10)
    print(f"  Class centers: {class_centers.shape}")

    # ------------------------------------------------------------------
    # 3. Load all events & compute embeddings
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Loading {args.max_events:,} events & computing embeddings")
    print("=" * 60)

    t0 = time.time()
    events, labels, weights = load_awkward(TEST_H5, max_events=args.max_events, lazy=False)
    labels = labels.numpy() if hasattr(labels, 'numpy') else np.array(labels)
    print(f"  Loaded {len(labels)} events in {time.time()-t0:.0f}s")

    t0 = time.time()
    embeddings, preds, _ = compute_embeddings_and_labels(model, events, labels, weights, device)
    print(f"  Embeddings: {embeddings.shape}, time: {time.time()-t0:.0f}s")

    emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)

    # Cosine-similarity decision scores & confusion
    t_sims = np.dot(emb_norm, centers_norm.T)
    preds = np.argmax(t_sims, axis=1)
    decision_scores = t_sims[:, 1] - t_sims[:, 0]
    cm = confusion_matrix(labels, preds)
    acc = np.trace(cm) / cm.sum()
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Confusion matrix:\n{cm}")

    # Freeze events memory; we will reload for physics later
    del model
    gc.collect()

    # ------------------------------------------------------------------
    # 4. Stratified 50 k subsample for UMAP
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Stratified sampling ({UMAP_SAMPLE:,}) for UMAP")
    print("=" * 60)

    np.random.seed(42)
    sub_idx = stratified_sample(labels, UMAP_SAMPLE // 2)
    sub_emb = emb_norm[sub_idx]
    sub_labels = labels[sub_idx]
    sub_preds = preds[sub_idx]
    sub_scores = decision_scores[sub_idx]

    # Prediction status for the subset
    tp = (sub_preds == 1) & (sub_labels == 1)
    tn = (sub_preds == 0) & (sub_labels == 0)
    fp = (sub_preds == 1) & (sub_labels == 0)
    fn = (sub_preds == 0) & (sub_labels == 1)

    status = np.full(len(sub_emb), '', dtype=object)
    status[tp] = 'True Positive'
    status[tn] = 'True Negative'
    status[fp] = 'False Positive'
    status[fn] = 'False Negative'

    print(f"  TP={tp.sum()}  TN={tn.sum()}  FP={fp.sum()}  FN={fn.sum()}")

    # ------------------------------------------------------------------
    # 5. UMAP (2-D, cosine metric)
    # ------------------------------------------------------------------
    print("\nFitting UMAP (2-D, cosine)...")
    import umap
    t0 = time.time()
    reducer = umap.UMAP(n_components=2, metric='cosine', random_state=42,
                        n_neighbors=15, min_dist=0.1, n_epochs=200, n_jobs=-1)
    umap_coords = reducer.fit_transform(sub_emb)
    centers_umap = reducer.transform(centers_norm)
    print(f"  Done in {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    # 6. Spectral clustering per class (predicted primary: what model sees)
    # ------------------------------------------------------------------
    part_labels = sub_preds if args.partition == "predicted" else sub_labels
    print(f"\nSpectral clustering on {args.partition} classes (partition={args.partition})...")
    subclass_labels = np.full(len(sub_emb), -1, dtype=int)

    top_mask = part_labels == 1
    if top_mask.sum() > 0:
        top_subclasses, k_top = spectral_clustering_subclass(sub_emb[top_mask], max_k=5)
        active_top, counts_top = np.unique(top_subclasses, return_counts=True)
        subclass_labels[top_mask] = top_subclasses + 1  # reserve 0 for unassigned
        print(f"  Top subclasses: {dict(zip(active_top, counts_top))}")
    else:
        active_top = np.array([], dtype=int)

    qcd_mask = part_labels == 0
    if qcd_mask.sum() > 0:
        qcd_subclasses, k_qcd = spectral_clustering_subclass(sub_emb[qcd_mask], max_k=5)
        active_qcd, counts_qcd = np.unique(qcd_subclasses, return_counts=True)
        subclass_labels[qcd_mask] = -(qcd_subclasses + 1)  # negative = QCD
        print(f"  QCD subclasses: {dict(zip(active_qcd, counts_qcd))}")
    else:
        active_qcd = np.array([], dtype=int)

    # Clean up subclass labels: renumber 0..K-1
    # QCD -> 0..Kq-1, Top -> Kq..Kq+Kt-1
    offset = 0
    sc_map = np.full(len(sub_emb), -1, dtype=int)
    if len(active_qcd) > 0:
        for i, sc in enumerate(active_qcd):
            sc_map[qcd_mask & (subclass_labels == -(sc + 1))] = offset + i
        offset += len(active_qcd)
    if len(active_top) > 0:
        for i, sc in enumerate(active_top):
            sc_map[top_mask & (subclass_labels == (sc + 1))] = offset + i
    subclass_labels = sc_map
    n_subclasses = offset + len(active_top)

    print(f"  Total subclasses: {n_subclasses}")

    # ------------------------------------------------------------------
    # 7. Plot 1: UMAP prediction status (2-D)
    # ------------------------------------------------------------------
    print("\nGenerating 2D UMAP Prediction Status plot...")
    fig, ax = plt.subplots(figsize=(8, 6.5))
    colors = {'True Negative': COLORS["qcd"], 'True Positive': COLORS["top"],
              'False Positive': COLORS["top_edge"], 'False Negative': COLORS["orange"]}
    markers = {'True Negative': 'o', 'True Positive': 'o',
               'False Positive': '^', 'False Negative': '^'}
    sizes = {'True Negative': 3, 'True Positive': 3,
             'False Positive': 20, 'False Negative': 20}
    alphas = {'True Negative': 0.12, 'True Positive': 0.12,
              'False Positive': 0.7, 'False Negative': 0.7}

    for cat in ['True Negative', 'True Positive', 'False Positive', 'False Negative']:
        mask = status == cat
        if mask.sum() > 0:
            ax.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                       c=colors[cat], marker=markers[cat], s=sizes[cat],
                       alpha=alphas[cat], label=f"{cat} ({mask.sum()})", edgecolors='none')

    for lbl, idx, color in [('W_QCD', 0, 'yellow'), ('W_Top', 1, 'magenta')]:
        ax.scatter(centers_umap[idx, 0], centers_umap[idx, 1],
                   marker='*', s=400, c=color, edgecolors='black', linewidth=0.8,
                   label=lbl, zorder=10)

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    panel_label(ax, "(a)", loc="left")
    ax.legend(loc='upper right', fontsize=7, **{k: v for k, v in LEGEND_KWARGS.items() if k not in ("ncol", "fontsize")})
    add_grid(ax)
    x_pad = (umap_coords[:, 0].max() - umap_coords[:, 0].min()) * 0.05
    y_pad = (umap_coords[:, 1].max() - umap_coords[:, 1].min()) * 0.05
    ax.set_xlim(umap_coords[:, 0].min() - x_pad, umap_coords[:, 0].max() + x_pad)
    ax.set_ylim(umap_coords[:, 1].min() - y_pad, umap_coords[:, 1].max() + y_pad)
    save_fig(fig, "umap_prediction_status.png", to_paper=True)

    # ------------------------------------------------------------------
    # 8. Plot 2: UMAP subclasses (2-D)
    # ------------------------------------------------------------------
    print("Generating 2D UMAP Subclasses plot...")
    fig, ax = plt.subplots(figsize=(8, 6.5))

    qcd_colors = ['#a1d99b', '#74c476', '#41ab5d', '#238b45', '#005a32']
    top_colors = ['#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#084594']

    for i in range(len(active_qcd)):
        mask = (subclass_labels == i)
        if mask.sum() > 0:
            ax.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                       c=qcd_colors[i % len(qcd_colors)], s=3, alpha=0.15,
                       label=f"QCD Subclass {i} (N={mask.sum()})", edgecolors='none')

    for i in range(len(active_top)):
        mask = (subclass_labels == (len(active_qcd) + i))
        if mask.sum() > 0:
            ax.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                       c=top_colors[i % len(top_colors)], s=3, alpha=0.15,
                       label=f"Top Subclass {i} (N={mask.sum()})", edgecolors='none')

    for lbl, idx, color in [('W_QCD', 0, 'yellow'), ('W_Top', 1, 'magenta')]:
        ax.scatter(centers_umap[idx, 0], centers_umap[idx, 1],
                   marker='*', s=400, c=color, edgecolors='black', linewidth=0.8,
                   label=lbl, zorder=10)

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    panel_label(ax, "(b)", loc="left")
    ax.legend(loc='upper right', fontsize=7, **{k: v for k, v in LEGEND_KWARGS.items() if k not in ("ncol", "fontsize")})
    add_grid(ax)
    ax.set_xlim(umap_coords[:, 0].min() - x_pad, umap_coords[:, 0].max() + x_pad)
    ax.set_ylim(umap_coords[:, 1].min() - y_pad, umap_coords[:, 1].max() + y_pad)
    save_fig(fig, "umap_subclasses.png", to_paper=True)

    # ------------------------------------------------------------------
    # 9. Stratified ~15 k sample for physics + Lund planes
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Stratified sampling for physics features ({PHYS_SAMPLE:,})")
    print("=" * 60)

    np.random.seed(42)
    phys_idx = []

    # Per-subclass sampling (up to 2000 per subclass)
    for sc in range(n_subclasses):
        sc_idx = np.where(subclass_labels == sc)[0]
        if len(sc_idx) > 0:
            n = min(2000, len(sc_idx))
            phys_idx.extend(np.random.choice(sc_idx, n, replace=False))

    # Per-status sampling (up to 2000 per status)
    for cat in ['True Positive', 'True Negative', 'False Positive', 'False Negative']:
        cat_idx = np.where(status == cat)[0]
        if len(cat_idx) > 0:
            n = min(2000, len(cat_idx))
            phys_idx.extend(np.random.choice(cat_idx, n, replace=False))

    phys_idx = np.unique(phys_idx)
    # Cap total to PHYS_SAMPLE
    if len(phys_idx) > PHYS_SAMPLE:
        phys_idx = np.random.choice(phys_idx, PHYS_SAMPLE, replace=False)
    print(f"  Selected {len(phys_idx)} jets for physics computation.")

    # Map back to global indices
    global_phys_idx = sub_idx[phys_idx]

    # Load the corresponding events from H5
    from arcefn.data.loader import load_events_by_indices
    phys_events = load_events_by_indices(TEST_H5, global_phys_idx)

    phys_labels = sub_labels[phys_idx]
    phys_scores = sub_scores[phys_idx]
    phys_status = status[phys_idx]
    phys_preds = sub_preds[phys_idx]
    phys_subclass = subclass_labels[phys_idx]

    # Physics features via FastJet
    print("Computing FastJet substructure features...")
    t0 = time.time()
    phys_features = compute_features(phys_events)
    print(f"  Done in {time.time()-t0:.0f}s, shape={phys_features.shape}")

    mass_all = phys_features[:, 0]
    tau32_all = phys_features[:, 7]
    d12_all = phys_features[:, 4]
    feature_names = ["Mass", "mSD", "Mult", "nSD", "sqrt(d12)", "sqrt(d23)",
                     "Tau21", "Tau32", "zg", "theta_g"]

    # ------------------------------------------------------------------
    # 10. Plot 3: Subclass physics distributions (KDE)
    # ------------------------------------------------------------------
    print("Generating subclass physics distributions KDE plots...")
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    panel_labels_grid = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)", "(i)"]
    feature_labels = ["Jet mass (GeV)", r"$\tau_{32}$", r"$\sqrt{d_{12}}$ (GeV)"]
    row_labels_grid = ["Jet mass", r"$\tau_{32}$", r"$\sqrt{d_{12}}$ (GeV)"]
    sc_qcd_colors = [COLORS["qcd"], COLORS["qcd_edge"]]
    sc_top_colors = [COLORS["top"], COLORS["top_edge"]]
    tp_color = COLORS["qcd"]
    fn_color = COLORS["orange"]
    tn_color = COLORS["green"]
    fp_color = COLORS["top"]

    for i in range(len(active_top)):
        sc = len(active_qcd) + i
        mask = phys_subclass == sc
        if mask.sum() > 5:
            sns.kdeplot(mass_all[mask], ax=axes[0, 0], fill=True, alpha=0.35,
                        color=sc_top_colors[i], label=f"Top subclass {i}")
            sns.kdeplot(tau32_all[mask], ax=axes[1, 0], fill=True, alpha=0.35,
                        color=sc_top_colors[i], label=f"Top subclass {i}")
            sns.kdeplot(d12_all[mask], ax=axes[2, 0], fill=True, alpha=0.35,
                        color=sc_top_colors[i], label=f"Top subclass {i}")

    tn_sub = phys_status == 'True Negative'
    fp_sub = phys_status == 'False Positive'
    if tn_sub.sum() > 5:
        sns.kdeplot(mass_all[tn_sub], ax=axes[0, 1], fill=True, alpha=0.3, color=tn_color, label="TN (QCD)")
        sns.kdeplot(tau32_all[tn_sub], ax=axes[1, 1], fill=True, alpha=0.3, color=tn_color, label="TN (QCD)")
        sns.kdeplot(d12_all[tn_sub], ax=axes[2, 1], fill=True, alpha=0.3, color=tn_color, label="TN (QCD)")
    if fp_sub.sum() > 5:
        sns.kdeplot(mass_all[fp_sub], ax=axes[0, 1], fill=False, linewidth=1.2, color=fp_color, label="FP (QCD→Top)")
        sns.kdeplot(tau32_all[fp_sub], ax=axes[1, 1], fill=False, linewidth=1.2, color=fp_color, label="FP (QCD→Top)")
        sns.kdeplot(d12_all[fp_sub], ax=axes[2, 1], fill=False, linewidth=1.2, color=fp_color, label="FP (QCD→Top)")

    tp_sub = phys_status == 'True Positive'
    fn_sub = phys_status == 'False Negative'
    if tp_sub.sum() > 5:
        sns.kdeplot(mass_all[tp_sub], ax=axes[0, 2], fill=True, alpha=0.3, color=tp_color, label="TP (Top)")
        sns.kdeplot(tau32_all[tp_sub], ax=axes[1, 2], fill=True, alpha=0.3, color=tp_color, label="TP (Top)")
        sns.kdeplot(d12_all[tp_sub], ax=axes[2, 2], fill=True, alpha=0.3, color=tp_color, label="TP (Top)")
    if fn_sub.sum() > 5:
        sns.kdeplot(mass_all[fn_sub], ax=axes[0, 2], fill=False, linewidth=1.2, color=fn_color, label="FN (Top→QCD)")
        sns.kdeplot(tau32_all[fn_sub], ax=axes[1, 2], fill=False, linewidth=1.2, color=fn_color, label="FN (Top→QCD)")
        sns.kdeplot(d12_all[fn_sub], ax=axes[2, 2], fill=False, linewidth=1.2, color=fn_color, label="FN (Top→QCD)")

    col_titles = ["Top subclasses", "QCD: TN vs FP", "Top: TP vs FN"]
    for j in range(3):
        axes[0, j].set_title(col_titles[j], fontsize=9)
        for i in range(3):
            panel_label(axes[i, j], panel_labels_grid[i * 3 + j], fontsize=9, pad=4)
            axes[i, j].set_xlabel(feature_labels[i] if i == 2 else "")
            axes[i, j].set_ylabel("Density" if j == 0 else "")
            axes[i, j].legend(fontsize=6.5, loc="upper right",
                              frameon=True, fancybox=True, framealpha=0.85, edgecolor="0.7")
            add_grid(axes[i, j])

    plt.tight_layout(w_pad=1.0, h_pad=0.8)
    save_fig(fig, "subclass_physics_distributions.png", to_paper=True)

    # ------------------------------------------------------------------
    # 11. Plot 4: Subclass composition heatmap
    # ------------------------------------------------------------------
    print("Generating subclass composition heatmap...")
    categories = ['True Positive', 'False Positive', 'True Negative', 'False Negative']
    composition_matrix = []
    row_labels = []

    for sc in range(n_subclasses):
        sc_mask = phys_subclass == sc
        n_tot = sc_mask.sum()
        if n_tot > 0:
            row_counts = []
            for cat in categories:
                cat_count = ((phys_status == cat) & sc_mask).sum()
                row_counts.append(cat_count / n_tot * 100.0)
            composition_matrix.append(row_counts)
            label = f"{'Top' if sc >= len(active_qcd) else 'QCD'} Subclass {sc % max(len(active_qcd), 1) if len(active_qcd) > 0 else sc} (N={n_tot})"
            row_labels.append(label)

    if len(composition_matrix) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(composition_matrix, annot=True, fmt=".1f", ax=ax,
                    xticklabels=categories, yticklabels=row_labels,
                    cmap="YlGnBu", cbar_kws={'label': 'Percentage (%)', 'shrink': 0.8},
                    annot_kws={"fontsize": 8})
        ax.set_xlabel("Prediction status")
        ax.set_ylabel("Discovered subclasses")
        plt.tight_layout()
        save_fig(fig, "subclass_composition_heatmap.png", to_paper=True)

    # ------------------------------------------------------------------
    # 12. Plot 5: Lund planes (TP / TN / FP / FN)
    # ------------------------------------------------------------------
    print("Generating Lund Plane plots...")
    for cat, fname in [('True Positive', 'status_tp.png'),
                       ('True Negative', 'status_tn.png'),
                       ('False Positive', 'status_fp.png'),
                       ('False Negative', 'status_fn.png')]:
        mask = phys_status == cat
        if mask.sum() > 5:
            plot_lund_plane(phys_events[mask], label=f"{cat} Jets",
                            save_path=os.path.join(lund_dir, fname))
            print(f"  Saved: lund_planes/{fname}")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"[COMPLETE] Paper figures saved to '{model_dir}'")
    print("=" * 60)


if __name__ == '__main__':
    main()
