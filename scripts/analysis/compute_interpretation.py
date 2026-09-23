"""
Interpretation Analysis (Full Dataset)
========================================
Uses all ~404K test events for robust embedding interpretability
statistics.  Processes physics features in chunks to manage memory.

Steps
-----
1.  Load all embeddings from the trained model.
2.  Compute physics features (chunked FastJet).
3.  Compute mutual information between embedding dims and physics.
4.  Compute Cohen's *d* effect sizes for subclass splits.
5.  PCA on the embedding space.
6.  Generate figures and save results JSON.

Output
------
* ``experiments/interpretation_full_mi.png``      — MI heatmap
* ``experiments/interpretation_full_effects.png`` — Effect-size bar charts
* ``experiments/interpretation_full_pca.png``     — PCA variance plot
* ``experiments/interpretation_full_results.json``
"""

import json
import time
import gc
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from arcefn.utils.paths import EXPERIMENTS, DATA_DIR, MODEL_DIR, CHECKPOINT
from arcefn.models.efn_arcface import TopTaggingModel
from arcefn.data.loader import load_awkward, JetTaggingDataset
from arcefn.utils.device import get_device
from arcefn.utils.physics import compute_features
from arcefn.utils.clustering import spectral_clustering_subclass
from arcefn.utils.figure_style import (
    apply_style, panel_label, add_grid, save_fig, COLORS,
)
apply_style()

TEST_H5 = str(DATA_DIR / 'test.h5')
RESULTS_PATH = str(EXPERIMENTS / 'interpretation_full_results.json')

FEATURE_NAMES = ["Mass", "mSD", "Mult", "nSD", "sqrt(d12)", "sqrt(d23)",
                 "Tau21", "Tau32", "zg", "theta_g"]


def load_model(path, device):
    """
    Load a model checkpoint with automatic architecture detection.

    Parameters
    ----------
    path : str or Path
        Path to the ``.pt`` checkpoint.
    device : torch.device
        Device to place the model on.

    Returns
    -------
    torch.nn.Module
        Loaded model in evaluation mode.
    """
    import os
    import json
    sd = torch.load(str(path), map_location='cpu', weights_only=False)
    
    # Try to infer embedding_dim from checkpoint
    if 'arcface_head.class_centers' in sd:
        embedding_dim = sd['arcface_head.class_centers'].shape[1]
    elif 'classifier.weight' in sd:
        embedding_dim = sd['classifier.weight'].shape[1]
    else:
        embedding_dim = 64  # fallback
    
    if 'arcface_head.class_centers' in sd:
        model = TopTaggingModel(s=16.0, m=0.5, embedding_dim=embedding_dim, particle_dim=128).to(device)
    elif 'classifier.weight' in sd:
        from arcefn.models.efn_linear import TopTaggingLinear
        model = TopTaggingLinear(embedding_dim=embedding_dim, particle_dim=128).to(device)
    else:
        from arcefn.models.efn_coslinear import TopTaggingCosLinear
        model = TopTaggingCosLinear(embedding_dim=embedding_dim, particle_dim=128, s=16.0).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def get_all_embeddings(model, h5_path, device):
    """
    Run inference on *all* test events and return embeddings, predictions,
    and labels.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model.
    h5_path : str
        Path to ``test.h5``.
    device : torch.device
        Compute device.

    Returns
    -------
    embeddings : ndarray, shape (N, D)
    preds : ndarray, shape (N,)
    labels : ndarray, shape (N,)
    """
    events, labels, weights = load_awkward(h5_path, max_events=999999)
    ds = JetTaggingDataset(events, labels, weights)
    loader = DataLoader(ds, batch_size=2048, shuffle=False, num_workers=0)

    all_emb, all_preds, all_labels = [], [], []
    n_total = len(labels)
    n_processed = 0

    with torch.no_grad():
        for x, y, w, m, _ in loader:
            out = model(x.to(device), mask=m.to(device))
            if isinstance(out, (tuple, list)):
                logits, emb = out[0], out[-1]
            else:
                logits, emb = out, out

            all_emb.append(emb.cpu().numpy())
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_labels.append(y.numpy())

            n_processed += len(y)
            if n_processed % 50000 == 0:
                print(f"    Processed {n_processed}/{n_total} ({100*n_processed//n_total}%)")

    return np.vstack(all_emb), np.concatenate(all_preds), np.concatenate(all_labels)


def get_all_physics_features(h5_path, chunk_size=5000):
    """
    Compute physics features for all test events in chunks.

    Falls back to basic (mass, multiplicity) if FastJet fails on a chunk.

    Parameters
    ----------
    h5_path : str
        Path to ``test.h5``.
    chunk_size : int
        Number of events per chunk.

    Returns
    -------
    features : ndarray, shape (N, 10)
        Physics feature matrix.
    labels : ndarray, shape (N,)
        Ground-truth labels.
    """
    events, labels, weights = load_awkward(h5_path, max_events=999999, lazy=False)
    n_total = len(events)
    all_features = []

    n_chunks = (n_total + chunk_size - 1) // chunk_size
    for i in range(n_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, n_total)
        chunk = events[start:end]

        try:
            features = compute_features(chunk)
            all_features.append(features)
        except Exception as e:
            print(f"    Chunk {i+1}/{n_chunks} failed: {e}")
            basic = np.zeros((len(chunk), 10))
            for j in range(len(chunk)):
                E = chunk[j][:, 0]
                px, py, pz = chunk[j][:, 1], chunk[j][:, 2], chunk[j][:, 3]
                jet_E, jet_px, jet_py, jet_pz = E.sum(), px.sum(), py.sum(), pz.sum()
                basic[j, 0] = np.sqrt(max(0, jet_E**2 - jet_px**2 - jet_py**2 - jet_pz**2))
                basic[j, 2] = (E > 0).sum()
            all_features.append(basic)

        if (i + 1) % 10 == 0:
            print(f"    Chunk {i+1}/{n_chunks} ({end}/{n_total})")

    return np.vstack(all_features), labels


def compute_mutual_information_full(embeddings, physics_features, labels, n_bins=15):
    """
    Compute pairwise mutual information between embedding dimensions and
    physics features on a 50 k subset.

    Parameters
    ----------
    embeddings : ndarray, shape (N, D)
    physics_features : ndarray, shape (N, P)
    labels : ndarray, shape (N,)
        Ignored; kept for API compatibility.
    n_bins : int
        Number of quantile bins for discretisation.

    Returns
    -------
    mi_matrix : ndarray, shape (D, P)
        MI values.
    """
    n_sample = len(embeddings)
    np.random.seed(42)
    idx = np.random.choice(len(embeddings), n_sample, replace=False)

    emb_sub = embeddings[idx]
    phys_sub = physics_features[idx]

    emb_disc = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    phys_disc = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')

    emb_d = emb_disc.fit_transform(emb_sub).astype(int)
    phys_d = phys_disc.fit_transform(phys_sub).astype(int)

    n_emb = embeddings.shape[1]
    n_phys = physics_features.shape[1]
    mi_matrix = np.zeros((n_emb, n_phys))
    for i in range(n_emb):
        for j in range(n_phys):
            mi_matrix[i, j] = mutual_info_score(emb_d[:, i], phys_d[:, j])

    return mi_matrix


def main():
    """
    Run the full interpretation pipeline on all ~404K test events.

    Loads the main model, computes all embeddings, physics features,
    mutual information, effect sizes, and PCA, then saves figures
    and a results JSON.
    """
    device, _ = get_device()
    print(f"Device: {device}\n")

    # Step 1: Load ALL embeddings
    print("=" * 60)
    print("Step 1: Loading ALL embeddings from full test set")
    print("=" * 60)

    model_path = str(CHECKPOINT)
    model = load_model(model_path, device)

    t0 = time.time()
    embeddings, preds, labels = get_all_embeddings(model, TEST_H5, device)
    print(f"  Total: {embeddings.shape}, Time: {time.time()-t0:.0f}s")

    emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
    acc = np.mean(preds == labels)
    print(f"  Accuracy: {acc:.4f}")

    del model
    gc.collect()

    # Step 2: Compute ALL physics features
    print(f"\n{'='*60}")
    print("Step 2: Computing ALL physics features (chunked)")
    print("=" * 60)

    t0 = time.time()
    physics_features, _ = get_all_physics_features(TEST_H5, chunk_size=5000)
    print(f"  Total: {physics_features.shape}, Time: {time.time()-t0:.0f}s")

    # Step 3: Mutual Information
    print(f"\n{'='*60}")
    print("Step 3: Mutual Information analysis")
    print("=" * 60)

    t0 = time.time()
    mi_matrix = compute_mutual_information_full(emb_norm, physics_features, labels)
    print(f"  MI matrix: {mi_matrix.shape}, Time: {time.time()-t0:.0f}s")

    mi_flat = []
    for i in range(mi_matrix.shape[0]):
        for j in range(mi_matrix.shape[1]):
            mi_flat.append((i, j, mi_matrix[i, j]))
    mi_flat.sort(key=lambda x: x[2], reverse=True)

    print("\n  Top 15 MI pairs (embedding dim <-> physics feature):")
    for rank, (e, p, m) in enumerate(mi_flat[:15]):
        print(f"    #{rank+1}: Emb {e:2d} <-> {FEATURE_NAMES[p]:12s}: MI={m:.4f}")

    # Step 4: Class correlation
    print(f"\n{'='*60}")
    print("Step 4: Class correlation (full dataset)")
    print("=" * 60)

    class_corr = np.array([abs(np.corrcoef(emb_norm[:, i], labels)[0, 1])
                           for i in range(emb_norm.shape[1])])
    top_dims = np.argsort(class_corr)[::-1][:15]

    print("  Top 15 class-discriminative embedding dimensions:")
    for rank, d in enumerate(top_dims):
        print(f"    #{rank+1}: Emb {d:2d}: |corr|={class_corr[d]:.4f}")

    # Step 5: Subclass analysis
    print(f"\n{'='*60}")
    print("Step 5: Subclass physics analysis (full dataset)")
    print("=" * 60)

    subclass_labels = np.full(len(labels), -1)
    for cls in [0, 1]:
        mask = labels == cls
        if mask.sum() > 10:
            sc_labels, _n_sc = spectral_clustering_subclass(emb_norm[mask], max_k=5)
            order = sorted(
                (c for c in np.unique(sc_labels) if c >= 0),
                key=lambda c: -int((sc_labels == c).sum()),
            )
            mapped = np.array([order.index(int(l)) if l >= 0 else -1 for l in sc_labels])
            subclass_labels[mask] = mapped + (2 * cls)

    effect_sizes = {}
    for i, name in enumerate(FEATURE_NAMES):
        if i >= physics_features.shape[1]:
            break
        m0 = subclass_labels == 0
        m1 = subclass_labels == 1
        m2 = subclass_labels == 2
        m3 = subclass_labels == 3

        d_qcd = 0.0
        if m0.sum() > 0 and m1.sum() > 0:
            d_qcd = (physics_features[m1, i].mean() - physics_features[m0, i].mean()) / (
                np.sqrt(physics_features[m0, i].var() + physics_features[m1, i].var()) + 1e-10)

        d_top = 0.0
        if m2.sum() > 0 and m3.sum() > 0:
            d_top = (physics_features[m3, i].mean() - physics_features[m2, i].mean()) / (
                np.sqrt(physics_features[m2, i].var() + physics_features[m3, i].var()) + 1e-10)

        qcd_mask = labels == 0
        top_mask = labels == 1
        d_overall = (physics_features[top_mask, i].mean() - physics_features[qcd_mask, i].mean()) / (
            np.sqrt(physics_features[qcd_mask, i].var() + physics_features[top_mask, i].var()) + 1e-10)

        effect_sizes[name] = {
            "d_qcd_subclass": float(d_qcd),
            "d_top_subclass": float(d_top),
            "d_overall": float(d_overall),
            "n_qcd_core": int(m0.sum()),
            "n_qcd_edge": int(m1.sum()),
            "n_top_core": int(m2.sum()),
            "n_top_edge": int(m3.sum()),
        }

    print("  Effect sizes (Cohen's d) with sample sizes:")
    print(f"  {'Feature':15s} {'d(QCD)':>10s} {'d(Top)':>10s} {'d(All)':>10s} "
          f"{'N(QCD)':>12s} {'N(Top)':>12s}")
    for name in FEATURE_NAMES:
        if name in effect_sizes:
            es = effect_sizes[name]
            print(f"  {name:15s} {es['d_qcd_subclass']:>10.3f} {es['d_top_subclass']:>10.3f} "
                  f"{es['d_overall']:>10.3f} "
                  f"{es['n_qcd_core']+es['n_qcd_edge']:>12d} "
                  f"{es['n_top_core']+es['n_top_edge']:>12d}")

    # Step 6: PCA
    print(f"\n{'='*60}")
    print("Step 6: PCA analysis (full dataset)")
    print("=" * 60)

    n_components = min(10, emb_norm.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(emb_norm)
    var_exp = pca.explained_variance_ratio_
    n_90 = int(np.argmax(np.cumsum(var_exp) >= 0.9) + 1) if len(var_exp) > 0 else 0
    n_99 = int(np.argmax(np.cumsum(var_exp) >= 0.99) + 1) if len(var_exp) > 0 else 0

    print(f"  PC1: {var_exp[0]:.4f}, PC2: {var_exp[1]:.4f}" + (f", PC3: {var_exp[2]:.4f}" if len(var_exp) > 2 else ""))
    print(f"  Components for 90% variance: {n_90}")
    print(f"  Components for 99% variance: {n_99}")

    # Step 7: Generate plots
    print(f"\n{'='*60}")
    print("Step 7: Generating plots")
    print("=" * 60)

    # Plot 1: MI heatmap
    fig, ax = plt.subplots(figsize=(9, 6))
    top_emb = np.argsort(class_corr)[::-1][:20]
    im = ax.imshow(mi_matrix[top_emb, :], aspect='auto', cmap='YlOrRd')
    ax.set_yticks(range(len(top_emb)))
    ax.set_yticklabels([f"Emb {d}" for d in top_emb], fontsize=7)
    ax.set_xticks(range(len(FEATURE_NAMES)))
    ax.set_xticklabels(FEATURE_NAMES, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Embedding dimension')
    ax.set_xlabel('Physics observable')
    plt.colorbar(im, label='Mutual information (nats)', shrink=0.85, pad=0.08)
    save_fig(fig, "interpretation_full_mi.png")

    # Plot 2: Effect sizes
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    for ax_idx, (title, key) in enumerate([
        ("QCD Subclasses (Core vs Edge)", "d_qcd_subclass"),
        ("Top Subclasses (Core vs Edge)", "d_top_subclass"),
        ("Overall (QCD vs Top)", "d_overall"),
    ]):
        ax = axes[ax_idx]
        features = [f for f in FEATURE_NAMES if f in effect_sizes]
        values = [effect_sizes[f][key] for f in features]
        colors = ['green' if abs(v) > 0.5 else 'orange' if abs(v) > 0.2 else 'red'
                  for v in values]
        ax.barh(features, values, color=colors)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.axvline(0.2, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(-0.2, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(-0.5, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel("Cohen's d (effect size)")
        ax.set_title(title)
        ax.grid(True, alpha=0.15, axis='x')
    plt.tight_layout()
    save_fig(plt.gcf(), "interpretation_full_effects.png")

    # Plot 3: PCA
    fig, ax = plt.subplots(figsize=(6, 4))
    n_show = min(10, len(var_exp))
    x = np.arange(n_show)
    ax.bar(x, var_exp[:n_show], color=COLORS["qcd"], alpha=0.6, width=0.6,
           edgecolor="white", linewidth=0.3, label='Individual')
    ax.plot(x, np.cumsum(var_exp[:n_show]), 'o-', color=COLORS["top"],
            markersize=4, linewidth=1, label='Cumulative')
    ax.axhline(0.9, color='0.4', linestyle='--', linewidth=0.6, label='90%')
    ax.axhline(0.99, color='0.2', linestyle=':', linewidth=0.6, label='99%')
    ax.set_xlabel('Principal component')
    ax.set_ylabel('Variance explained')
    ax.set_xticks(x)
    ax.set_xticklabels([str(i+1) for i in range(n_show)])
    ax.legend(loc="lower right", **{k: v for k, v in {
        "frameon": True, "fancybox": True, "framealpha": 0.92,
        "edgecolor": "0.45", "facecolor": "white", "fontsize": 7,
    }.items()})
    add_grid(ax, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    save_fig(fig, "interpretation_full_pca.png")

    # Save results
    results = {
        "n_events": int(len(labels)),
        "accuracy": float(acc),
        "mi_matrix": mi_matrix.tolist(),
        "top_mi_pairs": [
            {"emb_dim": int(e), "feature": FEATURE_NAMES[p], "mi": float(m)}
            for e, p, m in mi_flat[:15]
        ],
        "class_correlation": {int(i): float(c) for i, c in enumerate(class_corr)},
        "top_class_dims": [int(d) for d in top_dims],
        "effect_sizes": effect_sizes,
        "pca_variance": var_exp.tolist(),
        "n_components_90": int(n_90),
        "n_components_99": int(n_99),
    }

    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {RESULTS_PATH}")

    # Summary
    print(f"\n{'='*60}")
    print("FULL DATASET INTERPRETATION SUMMARY")
    print("=" * 60)
    print(f"  N events: {len(labels):,}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Top MI pair: Emb {mi_flat[0][0]} <-> {FEATURE_NAMES[mi_flat[0][1]]} "
          f"(MI={mi_flat[0][2]:.4f})")
    print(f"  Most discriminative dim: {top_dims[0]} (|corr|={class_corr[top_dims[0]]:.4f})")
    print(f"  PCA: {n_90} components for 90% variance, {n_99} for 99%")

    print("\n[COMPLETE] Full dataset interpretation analysis finished!")


if __name__ == '__main__':
    main()
