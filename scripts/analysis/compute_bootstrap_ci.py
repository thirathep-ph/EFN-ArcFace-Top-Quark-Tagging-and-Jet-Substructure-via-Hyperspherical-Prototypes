"""
Bootstrap Confidence Intervals, ROC Curves, and Subclass Stability
====================================================================
Computes bootstrap confidence intervals for model accuracy, ROC curves,
background rejection rates at fixed signal efficiency, and NMI/ARI
subclass stability matrices across five trained models.

Output
------
* ``experiments/final_roc_curves.png``     — ROC curves for all models
* ``experiments/final_rejection_rates.png`` — Background rejection at
  fixed signal efficiency
* ``experiments/final_nmi_ari.png``         — NMI / ARI heatmaps
* ``experiments/final_analysis_results.json`` — All numeric results
"""

import argparse
import json
import time
import gc
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from arcefn.utils.paths import EXPERIMENTS, DATA_DIR
from arcefn.utils.figure_style import (
    apply_style, panel_label, add_grid, save_fig,
    COLORS, LEGEND_KWARGS,
)
apply_style()
from arcefn.models.efn_arcface import TopTaggingModel
from arcefn.data.loader import load_awkward, get_h5_len
from arcefn.utils.device import get_device

TEST_H5 = str(DATA_DIR / 'test.h5')
RESULTS_PATH = str(EXPERIMENTS / 'final_analysis_results.json')


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
        from arcefn.models.efn_coslinear import TopTaggingCosLinear
        model = TopTaggingCosLinear(embedding_dim=embedding_dim, particle_dim=128, s=16.0).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def get_model_output(model, h5_path, device, max_events=100000, seed=42, chunk_size=100000):
    """
    Run inference and return probabilities, predictions, labels, embeddings.

    The full test set is processed in chunks. When ``max_events`` is smaller
    than the total number of events, a seeded random subsample (without
    replacement) is drawn once and the same events are used for every model,
    so that model comparisons remain strictly paired.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model.
    h5_path : str
        Path to ``test.h5``.
    device : torch.device
        Compute device.
    max_events : int
        Maximum number of events to use. If ``>=`` the total, all events are
        used; otherwise a seeded random subsample of this size is drawn.
    seed : int
        Random seed for the subsample.
    chunk_size : int
        Number of events loaded at a time from disk.

    Returns
    -------
    probs : ndarray, shape (N, 2)
        Softmax probabilities.
    preds : ndarray, shape (N,)
        Predicted class indices.
    labels : ndarray, shape (N,)
        Ground-truth class labels.
    embeddings : ndarray, shape (N, D)
        Embedding vectors.
    """
    total = get_h5_len(h5_path)
    if max_events is None or max_events >= total:
        index = None
        n_use = total
        print(f"  Using all {total:,} events")
    else:
        rng = np.random.default_rng(seed)
        index = np.sort(rng.choice(total, max_events, replace=False))
        n_use = len(index)
        print(f"  Using seeded random subsample of {n_use:,}/{total:,} events (seed={seed})")

    all_logits, all_emb, all_labels = [], [], []

    for c0 in range(0, total, chunk_size):
        c1 = min(c0 + chunk_size, total)
        events, labels_c, _ = load_awkward(h5_path, start_event=c0, max_events=c1 - c0, lazy=True)

        if index is not None:
            sel = index[(index >= c0) & (index < c1)] - c0
            if len(sel) == 0:
                continue
            E = np.asarray(events.E[sel], dtype=np.float32)
            pX = np.asarray(events.px[sel], dtype=np.float32)
            pY = np.asarray(events.py[sel], dtype=np.float32)
            pZ = np.asarray(events.pz[sel], dtype=np.float32)
            labels_c = labels_c[sel]
        else:
            E = np.asarray(events.E, dtype=np.float32)
            pX = np.asarray(events.px, dtype=np.float32)
            pY = np.asarray(events.py, dtype=np.float32)
            pZ = np.asarray(events.pz, dtype=np.float32)

        with torch.no_grad():
            for b0 in range(0, len(E), 1024):
                b1 = min(b0 + 1024, len(E))
                x = torch.from_numpy(np.stack([E[b0:b1], pX[b0:b1], pY[b0:b1], pZ[b0:b1]], axis=-1))
                m = (x[:, :, 0] > 0).float()
                out = model(x.to(device), mask=m.to(device))

                if isinstance(out, (tuple, list)):
                    logits = out[0]
                    emb = out[-1]
                else:
                    logits = out
                    emb = out

                all_logits.append(logits.cpu().numpy())
                all_emb.append(emb.cpu().numpy())
                all_labels.append(labels_c[b0:b1])

        del events, E, pX, pY, pZ
        gc.collect()

    logits = np.vstack(all_logits)
    emb = np.vstack(all_emb)
    labels = np.concatenate(all_labels)
    probs = F.softmax(torch.from_numpy(logits), dim=1).numpy()
    preds = logits.argmax(axis=1)

    return probs, preds, labels, emb


def compute_bootstrap_ci(probs, labels, n_bootstrap=1000, ci=0.95):
    """
    Compute bootstrap confidence intervals for accuracy.

    Parameters
    ----------
    probs : ndarray
        Softmax probabilities.
    labels : ndarray
        Ground-truth labels.
    n_bootstrap : int
        Number of bootstrap resamples.
    ci : float
        Confidence level (e.g. 0.95).

    Returns
    -------
    dict
        Contains ``mean``, ``std``, ``ci_lower``, ``ci_upper``.
    """
    n = len(labels)
    rng = np.random.RandomState(42)
    accs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        pred = probs[idx].argmax(axis=1)
        accs.append(np.mean(pred == labels[idx]))
    accs = np.array(accs)
    return {
        "mean": float(np.mean(accs)),
        "std": float(np.std(accs)),
        "ci_lower": float(np.percentile(accs, (1 - ci) / 2 * 100)),
        "ci_upper": float(np.percentile(accs, (1 + ci) / 2 * 100)),
    }


def compute_rejection_rates(probs, labels):
    """
    Compute ROC AUC and background rejection at fixed signal efficiencies.

    Parameters
    ----------
    probs : ndarray
        Softmax probabilities.
    labels : ndarray
        Ground-truth labels.

    Returns
    -------
    dict
        ``roc_auc`` and ``rejection_rates`` mapping TPR targets to
        FPR and rejection (1/FPR).
    """
    scores = probs[:, 1]
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc_val = auc(fpr, tpr)

    target_tprs = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
    rejection_rates = {}
    for target in target_tprs:
        idx = np.argmin(np.abs(tpr - target))
        rejection_rates[f"tpr_{int(target*100)}"] = {
            "tpr": float(tpr[idx]),
            "fpr": float(fpr[idx]),
            "rejection": float(1.0 / fpr[idx]) if fpr[idx] > 0 else None,
        }

    return {"roc_auc": float(roc_auc_val), "rejection_rates": rejection_rates}


def compute_nmi_across_models(embeddings_dict, labels_dict):
    """
    Compute NMI and ARI matrices from k-means subclass labels per model.

    Parameters
    ----------
    embeddings_dict : dict of str → ndarray
        Mapping from model name to embeddings.
    labels_dict : dict of str → ndarray
        Mapping from model name to ground-truth labels.

    Returns
    -------
    model_names : list of str
        Ordered model names.
    nmi_matrix : ndarray, shape (M, M)
        Pairwise NMI values.
    ari_matrix : ndarray, shape (M, M)
        Pairwise ARI values.
    """
    model_names = list(embeddings_dict.keys())
    n = len(model_names)

    subclass_labels = {}
    for name in model_names:
        emb = embeddings_dict[name]
        lab = labels_dict[name]

        all_sub = np.full(len(lab), -1)
        for cls in [0, 1]:
            mask = lab == cls
            if mask.sum() > 10:
                km = KMeans(n_clusters=2, random_state=42, n_init='auto').fit(emb[mask])
                all_sub[mask] = km.labels_ + (2 * cls)

        subclass_labels[name] = all_sub

    nmi_matrix = np.zeros((n, n))
    ari_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                nmi_matrix[i, j] = 1.0
                ari_matrix[i, j] = 1.0
            else:
                mask = (subclass_labels[model_names[i]] >= 0) & (subclass_labels[model_names[j]] >= 0)
                if mask.sum() > 0:
                    nmi_matrix[i, j] = normalized_mutual_info_score(
                        subclass_labels[model_names[i]][mask],
                        subclass_labels[model_names[j]][mask]
                    )
                    ari_matrix[i, j] = adjusted_rand_score(
                        subclass_labels[model_names[i]][mask],
                        subclass_labels[model_names[j]][mask]
                    )

    return model_names, nmi_matrix, ari_matrix


def main():
    """
    Run the full bootstrap + ROC + NMI/ARI analysis pipeline.

    Loads five models, computes bootstrap CIs, rejection rates, and
    subclass stability matrices, then saves figures and results JSON.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-events', type=int, default=100000,
                        help='Number of test events (seeded random subsample if < total; default 100000).')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for the subsample (default 42).')
    parser.add_argument('--model_dir', type=str, default="experiments/robustness_s16_m05",
                        help='Directory containing model checkpoint (for single model eval).')
    args = parser.parse_args()

    device, _ = get_device()
    print(f"Device: {device}\n")
    print(f"max_events={args.max_events}, seed={args.seed}, model_dir={args.model_dir}\n")

    # Part 1: Bootstrap CI + Rejection Rates
    print("=" * 60)
    print("Part 1: Bootstrap CI + Rejection Rates")
    print("=" * 60)

    model_configs = {
        "arcface_s16_m05": EXPERIMENTS / "robustness_s16_m05" / "checkpoint.pt",
        "linear_seed42": EXPERIMENTS / "baselines_linear_seed_42" / "checkpoint.pt",
        "coslinear_seed42": EXPERIMENTS / "baselines_coslinear_seed_42" / "checkpoint.pt",
    }

    # If model_dir is provided and different from default, add it
    if args.model_dir != "experiments/robustness_s16_m05":
        model_configs["d2_arcface"] = Path(args.model_dir) / "checkpoint.pt"

    all_probs, all_preds, all_labels_map = {}, {}, {}
    all_embeddings = {}
    bootstrap_results, rejection_results = {}, {}
    n_use = 0

    for name, path in model_configs.items():
        if not path.exists():
            print(f"  [SKIP] {name}")
            continue

        print(f"  [LOAD] {name}...", end=" ", flush=True)
        t0 = time.time()

        model = load_model(str(path), device)
        probs, preds, labels, emb = get_model_output(
            model, TEST_H5, device, max_events=args.max_events, seed=args.seed)
        n_use = len(labels)

        all_probs[name] = probs
        all_preds[name] = preds
        all_labels_map[name] = labels
        all_embeddings[name] = emb

        boot = compute_bootstrap_ci(probs, labels, n_bootstrap=1000)
        bootstrap_results[name] = boot

        rej = compute_rejection_rates(probs, labels)
        rejection_results[name] = rej

        print(f"Acc={np.mean(preds == labels):.4f}, "
              f"BootCI=[{boot['ci_lower']:.4f}, {boot['ci_upper']:.4f}], "
              f"AUC={rej['roc_auc']:.4f} ({time.time()-t0:.0f}s)")

        del model
        gc.collect()

    # Part 2: NMI/ARI across models
    print(f"\n{'='*60}")
    print("Part 2: NMI/ARI across models (subclass stability)")
    print("=" * 60)

    model_names, nmi_matrix, ari_matrix = compute_nmi_across_models(all_embeddings, all_labels_map)

    print(f"\n  NMI Matrix:")
    print(f"  {'':20s}", end="")
    for name in model_names:
        print(f"  {name[:12]:>12s}", end="")
    print()
    for i, name in enumerate(model_names):
        print(f"  {name:20s}", end="")
        for j in range(len(model_names)):
            print(f"  {nmi_matrix[i,j]:>12.3f}", end="")
        print()

    # Part 3: Generate plots
    print(f"\n{'='*60}")
    print("Part 3: Generating plots")
    print("=" * 60)

    # Plot 1: ROC curves with bootstrap CI shading
    fig, ax = plt.subplots(figsize=(7, 5.5))
    display_names = {
        "arcface_s16_m05": "EFN + ArcFace (ours)",
        "linear_seed42": "EFN + Linear",
        "coslinear_seed42": "EFN + CosLinear",
    }
    colors = {
        "arcface_s16_m05": COLORS["qcd"],
        "linear_seed42": COLORS["green"],
        "coslinear_seed42": COLORS["top"],
    }

    for name in model_names:
        scores = all_probs[name][:, 1]
        fpr, tpr, _ = roc_curve(all_labels_map[name], scores)
        roc_auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors.get(name, None),
                label=f"{display_names.get(name, name[:15])} (AUC={roc_auc_val:.3f})")

        # Bootstrap CI band for this model's ROC curve
        n = len(all_labels_map[name])
        rng = np.random.RandomState(42)
        tprs = []
        fpr_grid = np.linspace(0, 1, 200)
        for _ in range(200):
            idx = rng.choice(n, n, replace=True)
            sc = all_probs[name][idx, 1]
            lb = all_labels_map[name][idx]
            f_b, t_b, _ = roc_curve(lb, sc)
            tprs.append(np.interp(fpr_grid, f_b, t_b))
        tprs = np.array(tprs)
        mean_tpr = tprs.mean(axis=0)
        std_tpr = tprs.std(axis=0)
        ax.fill_between(fpr_grid, np.clip(mean_tpr - 1.96 * std_tpr, 0, 1),
                        np.clip(mean_tpr + 1.96 * std_tpr, 0, 1),
                        color=colors.get(name, None), alpha=0.15)

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=0.6)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    panel_label(ax, "(a)", loc="left")
    ax.legend(loc="lower right", **{k: v for k, v in LEGEND_KWARGS.items() if k != "ncol"})
    add_grid(ax)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    save_fig(fig, "final_roc_curves.png")

    # Plot 2: Rejection rates line curve
    fig, ax = plt.subplots(figsize=(7, 5.5))
    tpr_labels = ["tpr_30", "tpr_50", "tpr_70", "tpr_80", "tpr_90", "tpr_95"]
    tpr_values = [0.30, 0.50, 0.70, 0.80, 0.90, 0.95]

    for name in model_names:
        rejections = []
        for tpr_key in tpr_labels:
            r = rejection_results[name]["rejection_rates"].get(tpr_key, {})
            rejections.append(r.get("rejection", 0) or 0)
        ax.plot(tpr_values, rejections, 'o-', color=colors.get(name, None),
                label=display_names.get(name, name[:15]), linewidth=1.2, markersize=5)

    ax.set_xlabel('Signal Efficiency (TPR)')
    ax.set_ylabel('Background Rejection ($1/\\mathrm{FPR}$)')
    panel_label(ax, "(b)", loc="left")
    ax.set_xticks(tpr_values)
    ax.set_xticklabels([f'{int(v*100)}%' for v in tpr_values])
    ax.legend(loc="upper right", **{k: v for k, v in LEGEND_KWARGS.items() if k != "ncol"})
    add_grid(ax)
    ax.set_yscale('log')
    save_fig(fig, "final_rejection_rates.png")

    # Plot 3: NMI / ARI heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    short_names = [n[:12] for n in model_names]

    ax = axes[0]
    im = ax.imshow(nmi_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(model_names)))
    ax.set_yticks(range(len(model_names)))
    ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(short_names, fontsize=8)
    for i in range(len(model_names)):
        for j in range(len(model_names)):
            ax.text(j, i, f"{nmi_matrix[i,j]:.3f}", ha='center', va='center', fontsize=7,
                   color='white' if nmi_matrix[i, j] < 0.5 else 'black')
    plt.colorbar(im, ax=ax)
    ax.set_title('NMI: Subclass Stability')

    ax = axes[1]
    im = ax.imshow(ari_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(model_names)))
    ax.set_yticks(range(len(model_names)))
    ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(short_names, fontsize=8)
    for i in range(len(model_names)):
        for j in range(len(model_names)):
            ax.text(j, i, f"{ari_matrix[i,j]:.3f}", ha='center', va='center', fontsize=7,
                   color='white' if ari_matrix[i, j] < 0.5 else 'black')
    plt.colorbar(im, ax=ax)
    ax.set_title('ARI: Subclass Stability')

    plt.tight_layout()
    plt.savefig(str(EXPERIMENTS / "final_nmi_ari.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: final_nmi_ari.png")

    # Save results
    results = {
        "config": {
            "n_events": n_use,
            "seed": args.seed,
            "total_events": len(all_labels_map[model_names[0]]),
            "n_bootstrap": 1000,
        },
        "bootstrap_ci": bootstrap_results,
        "rejection_rates": rejection_results,
        "nmi_matrix": {"model_names": model_names, "matrix": nmi_matrix.tolist()},
        "ari_matrix": {"model_names": model_names, "matrix": ari_matrix.tolist()},
    }

    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {RESULTS_PATH}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("=" * 60)
    for name in model_names:
        b = bootstrap_results[name]
        r = rejection_results[name]
        print(f"  {name}:")
        print(f"    Bootstrap CI: [{b['ci_lower']:.4f}, {b['ci_upper']:.4f}] (mean={b['mean']:.4f})")
        print(f"    ROC AUC: {r['roc_auc']:.4f}")
        rej_50 = r['rejection_rates'].get('tpr_50', {}).get('rejection', None)
        rej_80 = r['rejection_rates'].get('tpr_80', {}).get('rejection', None)
        if rej_50:
            print(f"    Rejection@50%TPR: {rej_50:.1f}")
        if rej_80:
            print(f"    Rejection@80%TPR: {rej_80:.1f}")

    nmi_vals = [nmi_matrix[i, j] for i in range(len(model_names)) for j in range(i + 1, len(model_names))]
    ari_vals = [ari_matrix[i, j] for i in range(len(model_names)) for j in range(i + 1, len(model_names))]
    print(f"\n  NMI across models: mean={np.mean(nmi_vals):.3f}, std={np.std(nmi_vals):.3f}")
    print(f"  ARI across models: mean={np.mean(ari_vals):.3f}, std={np.std(ari_vals):.3f}")

    print("\n[COMPLETE] Final analysis finished!")


if __name__ == '__main__':
    main()
