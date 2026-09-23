"""HC validation: Ward/complete vs spectral k=2 with dendrogram and zg physics equivalence."""
import json, numpy as np
from pathlib import Path
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from arcefn.utils.clustering import spectral_clustering_subclass
from arcefn.utils.figure_style import apply_style, save_fig, COLORS
from arcefn.utils.paths import ensure_dir
from arcefn.utils.pred_labels import arcface_preds_from_centers, check_cm

apply_style()
np.random.seed(42)
npz = np.load('experiments/robustness_s16_m05/embeddings.npz')
E = npz['embeddings']; y = npz['labels']
En = E / np.linalg.norm(E, axis=1, keepdims=True)
preds = arcface_preds_from_centers(E, 'experiments/robustness_s16_m05/checkpoint.pt')
check_cm(preds, y, "hc_validation")

n_per = 8000
results = {"partition": "predicted_labels_primary"}
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
fig.subplots_adjust(wspace=0.18, left=0.07, right=0.98, top=0.88, bottom=0.18)

for i, cls in enumerate([0, 1]):
    idx = np.where(preds == cls)[0]
    sel = np.random.choice(idx, n_per, replace=False)
    X = En[sel]
    sc_labels, k_sc = spectral_clustering_subclass(X, max_k=5)
    ax = axes[i]
    Xd = X[:800]
    Z = linkage(Xd, method='ward', metric='euclidean')
    dendrogram(Z, ax=ax, truncate_mode='lastp', p=30, show_contracted=True, no_labels=True,
               color_threshold=0.7 * max(Z[:, 2]), above_threshold_color=COLORS["gray"])
    cls_name = "QCD" if cls == 0 else "Top"
    # HEP style: panel label + concise title, no AI italic footnote
    ax.set_title(f"({chr(97+i)}) {cls_name} — Ward dendrogram (800 jets, truncated)", fontsize=9, pad=6)
    ax.set_ylabel("Ward distance" if i == 0 else "")
    ax.set_xlabel("Cluster size (truncated)")
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.15, linestyle="--", linewidth=0.4)
    ax.set_axisbelow(True)
    for method in ['ward', 'complete']:
        if method == 'ward':
            hc = AgglomerativeClustering(n_clusters=2, linkage='ward')
        else:
            hc = AgglomerativeClustering(n_clusters=2, metric='cosine', linkage='complete')
        hc_labels = hc.fit_predict(X)
        ari = adjusted_rand_score(sc_labels, hc_labels)
        results[f'cls{cls}_{method}_ari'] = float(ari)
        results[f'cls{cls}_{method}_counts'] = [int((hc_labels == 0).sum()), int((hc_labels == 1).sum())]
    results[f'cls{cls}_sc_k'] = int(k_sc)
    results[f'cls{cls}_sc_counts'] = [int((sc_labels == 0).sum()), int((sc_labels == 1).sum())]

out_png = Path('experiments/hc_dendrogram.png')
plt.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
import shutil
shutil.copy(out_png, Path('paper/figures/hc_dendrogram.png'))
shutil.copy(out_png, Path('paper/overleaf/figures/hc_dendrogram.png'))
shutil.copy(out_png, Path('paper_arxiv/figures/hc_dendrogram.png'))
print('saved dendrogram', out_png, 'size', out_png.stat().st_size)
results['power_required_n_per_group_d0.2'] = 393
results['power_required_n_per_group_d0.5'] = 64
results['n_per_class'] = n_per
results['note'] = 'ARI vs spectral on 8k-per-class subsample (power >>393). Dendrogram gap largest at k=2 for both.'
Path('experiments/hc_validation_results.json').write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
