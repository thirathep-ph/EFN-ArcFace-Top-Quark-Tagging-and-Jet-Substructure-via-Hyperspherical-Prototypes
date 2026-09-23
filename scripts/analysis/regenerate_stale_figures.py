"""Regenerate 4 stale figures using full-test-set spectral partition.

Figures regenerated:
  1. umap_prediction_status.png  — full-set TP/TN/FP/FN counts
  2. umap_subclasses.png         — full-set spectral subclass counts
  3. subclass_composition_heatmap.png — full-set composition matrix
  4. angular_collimation.png     — full-set spectral counts + θg vs 2M/pT

Runs spectral clustering per PREDICTED class on full 404k embeddings.
UMAP is fitted on a 50k stratified subsample (cosine metric).
"""
from __future__ import annotations

import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from shutil import copy2

from arcefn.utils.paths import EXPERIMENTS, DATA_DIR
from arcefn.utils.clustering import spectral_clustering_subclass

EMBEDDINGS = EXPERIMENTS / "robustness_s16_m05" / "embeddings.npz"
FEATURES = EXPERIMENTS / "rf_cache_test.npz"
CHECKPOINT = EXPERIMENTS / "robustness_s16_m05" / "checkpoint.pt"
PT_CACHE = EXPERIMENTS / "pt_cache_test.npz"
FIGURES = EXPERIMENTS.parent / "paper" / "figures"
UMAP_SAMPLE = 50000
MAX_PLOT = 25000
SEED = 42


def _load_class_centers():
    import torch
    sd = torch.load(str(CHECKPOINT), map_location='cpu', weights_only=False)
    centers = sd['arcface_head.class_centers'].numpy()
    return centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-10)


def _recompute_predictions(emb_norm, centers_norm):
    sims = emb_norm @ centers_norm.T
    return np.argmax(sims, axis=1), sims[:, 1] - sims[:, 0]


def _stratified_sample(labels, n_per_class, rng):
    idx = []
    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        n = min(n_per_class, len(cls_idx))
        idx.append(rng.choice(cls_idx, n, replace=False))
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return idx


def _run_spectral_per_predicted_class(emb_norm, preds):
    """Run spectral clustering per predicted class on full 404k embeddings."""
    subclass_labels = np.full(len(preds), -1, dtype=int)
    qcd_mask = preds == 0
    top_mask = preds == 1

    if qcd_mask.sum() > 0:
        qcd_sc, k_qcd = spectral_clustering_subclass(emb_norm[qcd_mask], max_k=10)
        qcd_active, qcd_counts = np.unique(qcd_sc, return_counts=True)
        # Map to 0=Core, 1=Edge (Core=larger cluster)
        core_idx = np.argmax(qcd_counts)
        qcd_mapped = np.where(qcd_sc == qcd_active[core_idx], 0, 1)
        subclass_labels[qcd_mask] = qcd_mapped
        print(f"  QCD spectral: k={k_qcd} Core={int((qcd_mapped==0).sum()):,} Edge={int((qcd_mapped==1).sum()):,}")

    if top_mask.sum() > 0:
        top_sc, k_top = spectral_clustering_subclass(emb_norm[top_mask], max_k=10)
        top_active, top_counts = np.unique(top_sc, return_counts=True)
        core_idx = np.argmax(top_counts)
        top_mapped = np.where(top_sc == top_active[core_idx], 0, 1)
        subclass_labels[top_mask] = top_mapped
        print(f"  Top spectral: k={k_top} Core={int((top_mapped==0).sum()):,} Edge={int((top_mapped==1).sum()):,}")

    return subclass_labels


def main():
    t0 = time.time()
    rng = np.random.RandomState(SEED)

    # ── Load data ─────────────────────────────────────────────────────
    emb_data = np.load(EMBEDDINGS)
    embeddings = emb_data["embeddings"]       # (404k, 64)
    gt_labels = emb_data["labels"]            # (404k,)

    feat_data = np.load(FEATURES)
    features = feat_data["features"]          # (404k, 10)

    centers_norm = _load_class_centers()
    emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)

    preds, decision_scores = _recompute_predictions(emb_norm, centers_norm)
    print(f"Loaded {len(embeddings):,} events  (t={time.time()-t0:.0f}s)")

    # ── Run spectral clustering per predicted class on FULL 404k ──────
    print("Running spectral clustering per predicted class (full 404k)...")
    subclass_labels = _run_spectral_per_predicted_class(emb_norm, preds)

    # Full-set counts
    tp_full = int(((preds == 1) & (gt_labels == 1)).sum())
    tn_full = int(((preds == 0) & (gt_labels == 0)).sum())
    fp_full = int(((preds == 1) & (gt_labels == 0)).sum())
    fn_full = int(((preds == 0) & (gt_labels == 1)).sum())
    print(f"  Full test: TP={tp_full} TN={tn_full} FP={fp_full} FN={fn_full}")

    qcd_mask = preds == 0
    top_mask = preds == 1
    qcd_sub0 = int(((preds == 0) & (subclass_labels == 0)).sum())
    qcd_sub1 = int(((preds == 0) & (subclass_labels == 1)).sum())
    top_sub0 = int(((preds == 1) & (subclass_labels == 0)).sum())
    top_sub1 = int(((preds == 1) & (subclass_labels == 1)).sum())
    print(f"  Spectral: QCD_Core={qcd_sub0:,} QCD_Edge={qcd_sub1:,} "
          f"Top_Core={top_sub0:,} Top_Edge={top_sub1:,}")

    # ── UMAP on 50k subsample ────────────────────────────────────────
    print(f"\nStratified {UMAP_SAMPLE:,} subsample for UMAP...")
    sub_idx = _stratified_sample(gt_labels, UMAP_SAMPLE // 2, rng)
    sub_emb = emb_norm[sub_idx]
    sub_labels = gt_labels[sub_idx]
    sub_preds = preds[sub_idx]
    sub_subclass = subclass_labels[sub_idx]
    sub_scores = decision_scores[sub_idx]

    tp_sub = (sub_preds == 1) & (sub_labels == 1)
    tn_sub = (sub_preds == 0) & (sub_labels == 0)
    fp_sub = (sub_preds == 1) & (sub_labels == 0)
    fn_sub = (sub_preds == 0) & (sub_labels == 1)

    status = np.full(len(sub_emb), '', dtype=object)
    status[tp_sub] = 'True Positive'
    status[tn_sub] = 'True Negative'
    status[fp_sub] = 'False Positive'
    status[fn_sub] = 'False Negative'
    print(f"  Subsample: TP={tp_sub.sum()} TN={tn_sub.sum()} FP={fp_sub.sum()} FN={fn_sub.sum()}")

    # ── Fit UMAP ──────────────────────────────────────────────────────
    print("Fitting UMAP (2-D, cosine)...")
    import umap
    t1 = time.time()
    reducer = umap.UMAP(n_components=2, metric='cosine', random_state=SEED,
                        n_neighbors=15, min_dist=0.1, n_epochs=200, n_jobs=-1)
    umap_coords = reducer.fit_transform(sub_emb)
    centers_umap = reducer.transform(centers_norm)
    print(f"  UMAP done in {time.time()-t1:.1f}s")

    x_pad = (umap_coords[:, 0].max() - umap_coords[:, 0].min()) * 0.05
    y_pad = (umap_coords[:, 1].max() - umap_coords[:, 1].min()) * 0.05

    # ════════════════════════════════════════════════════════════════════
    # FIGURE 1: UMAP prediction status (full-set counts in legend)
    # ════════════════════════════════════════════════════════════════════
    print("\n[1/4] umap_prediction_status.png")
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = {'True Negative': '#2ca02c', 'True Positive': '#1f77b4',
              'False Positive': '#d62728', 'False Negative': '#ff7f0e'}
    markers = {'True Negative': 'o', 'True Positive': 'o',
               'False Positive': '^', 'False Negative': '^'}
    sizes = {'True Negative': 4, 'True Positive': 4,
             'False Positive': 30, 'False Negative': 30}
    alphas = {'True Negative': 0.15, 'True Positive': 0.15,
              'False Positive': 0.8, 'False Negative': 0.8}
    full_counts = {'True Positive': tp_full, 'True Negative': tn_full,
                   'False Positive': fp_full, 'False Negative': fn_full}

    for cat in ['True Negative', 'True Positive', 'False Positive', 'False Negative']:
        mask = status == cat
        if mask.sum() > 0:
            ax.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                       c=colors[cat], marker=markers[cat], s=sizes[cat],
                       alpha=alphas[cat],
                       label=f"{cat} ({full_counts[cat]:,})", edgecolors='none')

    for lbl, idx, color in [('W_QCD', 0, 'yellow'), ('W_Top', 1, 'magenta')]:
        ax.scatter(centers_umap[idx, 0], centers_umap[idx, 1],
                   marker='*', s=500, c=color, edgecolors='black', linewidth=1.5,
                   label=lbl, zorder=10)

    ax.set_title("EFN + ArcFace Latent Space (UMAP) & Prediction Status", fontsize=14)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)
    ax.grid(True, alpha=0.15)
    ax.set_xlim(umap_coords[:, 0].min() - x_pad, umap_coords[:, 0].max() + x_pad)
    ax.set_ylim(umap_coords[:, 1].min() - y_pad, umap_coords[:, 1].max() + y_pad)
    plt.tight_layout()
    plt.savefig(str(FIGURES / 'umap_prediction_status.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # ════════════════════════════════════════════════════════════════════
    # FIGURE 2: UMAP subclasses (full-set counts in legend)
    # ════════════════════════════════════════════════════════════════════
    print("[2/4] umap_subclasses.png")
    fig, ax = plt.subplots(figsize=(10, 8))

    qcd_colors = ['#a1d99b', '#74c476', '#41ab5d', '#238b45', '#005a32']
    top_colors = ['#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#084594']

    qcd_full = [qcd_sub0, qcd_sub1]
    for i in range(2):
        mask = (sub_subclass == i) & (sub_preds == 0)
        if mask.sum() > 0:
            ax.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                       c=qcd_colors[i], s=4, alpha=0.2,
                       label=f"QCD Subclass {i} (N={qcd_full[i]:,})", edgecolors='none')

    top_full = [top_sub0, top_sub1]
    for i in range(2):
        mask = (sub_subclass == i) & (sub_preds == 1)
        if mask.sum() > 0:
            ax.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                       c=top_colors[i], s=4, alpha=0.2,
                       label=f"Top Subclass {i} (N={top_full[i]:,})", edgecolors='none')

    for lbl, idx, color in [('W_QCD', 0, 'yellow'), ('W_Top', 1, 'magenta')]:
        ax.scatter(centers_umap[idx, 0], centers_umap[idx, 1],
                   marker='*', s=500, c=color, edgecolors='black', linewidth=1.5,
                   label=lbl, zorder=10)

    ax.set_title("Discovered Subclasses in Latent Space (UMAP + Spectral Clustering)",
                 fontsize=14)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)
    ax.grid(True, alpha=0.15)
    ax.set_xlim(umap_coords[:, 0].min() - x_pad, umap_coords[:, 0].max() + x_pad)
    ax.set_ylim(umap_coords[:, 1].min() - y_pad, umap_coords[:, 1].max() + y_pad)
    plt.tight_layout()
    plt.savefig(str(FIGURES / 'umap_subclasses.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # ════════════════════════════════════════════════════════════════════
    # FIGURE 3: Subclass composition heatmap (full 404k)
    # ════════════════════════════════════════════════════════════════════
    print("[3/4] subclass_composition_heatmap.png")
    categories = ['True Positive', 'False Positive', 'True Negative', 'False Negative']

    full_status = np.full(len(embeddings), '', dtype=object)
    full_status[(preds == 1) & (gt_labels == 1)] = 'True Positive'
    full_status[(preds == 0) & (gt_labels == 0)] = 'True Negative'
    full_status[(preds == 1) & (gt_labels == 0)] = 'False Positive'
    full_status[(preds == 0) & (gt_labels == 1)] = 'False Negative'

    composition_matrix = []
    row_labels = []
    full_sub_counts = [qcd_sub0, qcd_sub1, top_sub0, top_sub1]
    sub_names = ['QCD Core', 'QCD Edge', 'Top Core', 'Top Edge']

    for sc_idx, (sc_val, sc_name, sc_n) in enumerate(
            zip([0, 1, 0, 1], sub_names, full_sub_counts)):
        if sc_idx < 2:
            sc_mask = (preds == 0) & (subclass_labels == sc_val)
        else:
            sc_mask = (preds == 1) & (subclass_labels == sc_val)

        n_tot = int(sc_mask.sum())
        if n_tot > 0:
            row_counts = []
            for cat in categories:
                cat_count = int(((full_status == cat) & sc_mask).sum())
                row_counts.append(cat_count / n_tot * 100.0)
            composition_matrix.append(row_counts)
            row_labels.append(f"{sc_name} (N={n_tot:,})")

    plt.figure(figsize=(10, 6))
    sns.heatmap(composition_matrix, annot=True, fmt=".1f",
                xticklabels=categories, yticklabels=row_labels,
                cmap="YlGnBu", cbar_kws={'label': 'Percentage (%)'})
    plt.title("Subclass Classification Purity / Composition", fontsize=14)
    plt.xlabel("Prediction Status")
    plt.ylabel("Discovered Subclasses")
    plt.tight_layout()
    plt.savefig(str(FIGURES / 'subclass_composition_heatmap.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # ════════════════════════════════════════════════════════════════════
    # FIGURE 4: Angular collimation (full 404k)
    # ════════════════════════════════════════════════════════════════════
    print("[4/4] angular_collimation.png")
    mass = features[:, 0]
    theta_g = features[:, 9]

    if PT_CACHE.exists():
        print("  Loading cached pT...")
        pt = np.load(PT_CACHE)["pt"]
    else:
        print("  Computing pT from H5 (slow first run)...")
        events_path = str(DATA_DIR / 'test.h5')
        from arcefn.data.loader import load_awkward
        events, _, _ = load_awkward(events_path, max_events=len(embeddings), lazy=False)
        pt = np.sqrt(np.sum(events.px, axis=1)**2 + np.sum(events.py, axis=1)**2)
        pt = np.array(pt)
        np.savez_compressed(PT_CACHE, pt=pt)
        print(f"  Cached pT to {PT_CACHE}")
        del events

    two_m_pt = 2.0 * mass / np.clip(pt, 1.0, None)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    qcd_colors_ang = ['#a1d99b', '#74c476', '#41ab5d', '#238b45']
    top_colors_ang = ['#9ecae1', '#6baed6', '#4292c6', '#2171b5']

    rng_plot = np.random.RandomState(SEED)

    for i, (sc_val, full_n) in enumerate([(0, qcd_sub0), (1, qcd_sub1)]):
        mask = (preds == 0) & (subclass_labels == sc_val) & \
               (theta_g > 0.01) & (theta_g < 1.5) & \
               (two_m_pt > 0.01) & (two_m_pt < 1.5)
        n = int(mask.sum())
        idx_plot = rng_plot.choice(np.where(mask)[0], min(MAX_PLOT, n), replace=False)
        axes[0].scatter(two_m_pt[idx_plot], theta_g[idx_plot], alpha=0.15, s=3,
                        color=qcd_colors_ang[i],
                        label=f"QCD Subclass {sc_val} (N={full_n:,})")

    for i, (sc_val, full_n) in enumerate([(0, top_sub0), (1, top_sub1)]):
        mask = (preds == 1) & (subclass_labels == sc_val) & \
               (theta_g > 0.01) & (theta_g < 1.5) & \
               (two_m_pt > 0.01) & (two_m_pt < 1.5)
        n = int(mask.sum())
        idx_plot = rng_plot.choice(np.where(mask)[0], min(MAX_PLOT, n), replace=False)
        axes[1].scatter(two_m_pt[idx_plot], theta_g[idx_plot], alpha=0.15, s=3,
                        color=top_colors_ang[i],
                        label=f"Top Subclass {sc_val} (N={full_n:,})")

    for ax, lbl in [(axes[0], "QCD Subclasses"), (axes[1], "Top Subclasses")]:
        ax.plot([0, 1.5], [0, 1.5], 'k--', lw=1.5, label=r'$\Delta R = 2M/p_T$')
        ax.set_xlabel(r"$2M/p_T$ (predicted $\Delta R$)")
        ax.set_ylabel(r"$\theta_g$ (measured)")
        ax.set_title(lbl)
        ax.set_xlim(0, 1.2); ax.set_ylim(0, 1.5)
        ax.legend(fontsize=8, loc='lower right', framealpha=0.9)
        ax.grid(True, alpha=0.15)

    plt.suptitle(r"Relativistic Angular Scaling: $\theta_g \approx 2M/p_T$", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(str(FIGURES / 'angular_collimation.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # ── Summary ───────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"[DONE] 4 figures regenerated in {elapsed:.0f}s")
    print(f"  Saved to: {FIGURES}")
    for f in ['umap_prediction_status.png', 'umap_subclasses.png',
              'subclass_composition_heatmap.png', 'angular_collimation.png']:
        print(f"    - {f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
