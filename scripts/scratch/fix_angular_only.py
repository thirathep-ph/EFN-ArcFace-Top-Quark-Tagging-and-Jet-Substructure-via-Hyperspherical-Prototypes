"""Regenerate only angular_collimation.png with lower-right legend."""
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from arcefn.utils.paths import DATA_DIR, EXPERIMENTS
from arcefn.utils.clustering import spectral_clustering_subclass

SEED = 42
MAX_PLOT = 20000
FIGURES = EXPERIMENTS.parent / 'paper' / 'figures'
PT_CACHE = EXPERIMENTS / 'pt_cache_test.npz'


def main():
    t0 = time.time()
    print("Loading embeddings...")
    emb_data = np.load(EXPERIMENTS / 'robustness_s16_m05' / 'embeddings.npz')
    embeddings = emb_data['embeddings']
    print(f"  {embeddings.shape[0]:,} events")

    print("Computing predictions...")
    import torch
    sd = torch.load(str(EXPERIMENTS / 'robustness_s16_m05' / 'checkpoint.pt'),
                    map_location='cpu', weights_only=False)
    centers = sd['arcface_head.class_centers'].numpy()
    emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
    ctr_norm = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-10)
    sims = emb_norm @ ctr_norm.T
    preds = np.argmax(sims, axis=1)

    print("Running spectral clustering per predicted class...")
    subclass_labels = np.full(len(embeddings), -1, dtype=int)
    subclass_counts = {}
    for cls_val, cls_name in [(0, 'QCD'), (1, 'Top')]:
        mask = preds == cls_val
        X_cls = emb_norm[mask]
        sub, k = spectral_clustering_subclass(X_cls, max_k=10)
        active, counts = np.unique(sub, return_counts=True)
        core_idx = np.argmax(counts)
        mapped = np.where(sub == active[core_idx], 0, 1)
        subclass_labels[mask] = mapped
        for sv in range(2):
            n = int((mapped == sv).sum())
            subclass_counts[f"{cls_name}_sub{sv}"] = n
            print(f"  {cls_name} subclass {sv}: {n:,}")

    print("Loading pT and theta_g...")
    features = np.load(EXPERIMENTS / 'rf_cache_test.npz')['features']
    mass, theta_g = features[:, 0], features[:, 9]

    if PT_CACHE.exists():
        pt = np.load(PT_CACHE)["pt"]
    else:
        from arcefn.data.loader import load_awkward
        events, _, _ = load_awkward(str(DATA_DIR / 'test.h5'), max_events=len(embeddings), lazy=False)
        pt = np.sqrt(np.sum(events.px, axis=1)**2 + np.sum(events.py, axis=1)**2)
        pt = np.array(pt)
        np.savez_compressed(PT_CACHE, pt=pt)
        del events

    two_m_pt = 2.0 * mass / np.clip(pt, 1.0, None)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    qcd_colors = ['#a1d99b', '#74c476']
    top_colors = ['#9ecae1', '#6baed6']
    rng = np.random.RandomState(SEED)

    for i, (sv, fn) in enumerate([(0, subclass_counts['QCD_sub0']),
                                   (1, subclass_counts['QCD_sub1'])]):
        mask = (preds == 0) & (subclass_labels == sv) & (theta_g > 0.01) & (theta_g < 1.5) & (two_m_pt > 0.01) & (two_m_pt < 1.5)
        idx = rng.choice(np.where(mask)[0], min(MAX_PLOT, int(mask.sum())), replace=False)
        axes[0].scatter(two_m_pt[idx], theta_g[idx], alpha=0.15, s=3, color=qcd_colors[i],
                        label=f"QCD Subclass {sv} (N={fn:,})")

    for i, (sv, fn) in enumerate([(0, subclass_counts['Top_sub0']),
                                   (1, subclass_counts['Top_sub1'])]):
        mask = (preds == 1) & (subclass_labels == sv) & (theta_g > 0.01) & (theta_g < 1.5) & (two_m_pt > 0.01) & (two_m_pt < 1.5)
        idx = rng.choice(np.where(mask)[0], min(MAX_PLOT, int(mask.sum())), replace=False)
        axes[1].scatter(two_m_pt[idx], theta_g[idx], alpha=0.15, s=3, color=top_colors[i],
                        label=f"Top Subclass {sv} (N={fn:,})")

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
    print(f"  Saved to {FIGURES / 'angular_collimation.png'}")
    print(f"  Done in {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
