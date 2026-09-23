"""Identify the two dominant PCA axes of the embedding and their physics meaning.

Projects the L2-normalized 64-d embeddings onto their two principal components
(the same PCA reported by ``compute_interpretation.py``, which explains 99.99%
of variance) and correlates each axis with the ten FastJet observables to name
what each axis encodes. CPU-only: reads cached ``embeddings.npz`` +
``rf_cache_test.npz``, no model forward pass, no GPU.

Output: ``experiments/embedding_axes.json``.
"""
import json
import time

import numpy as np
from scipy import stats
from sklearn.decomposition import PCA

from arcefn.utils.paths import EXPERIMENTS, MODEL_DIR

FEATURE_NAMES = [
    "mass", "m_sd", "multiplicity", "n_sd", "sqrt_d12",
    "sqrt_d23", "tau21", "tau32", "zg", "theta_g",
]


def _r2(y, X):
    A = np.column_stack([np.ones(len(y)), X])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ coef
    return 1.0 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)


def main():
    t0 = time.time()
    emb = np.load(MODEL_DIR / "embeddings.npz")
    X = emb["embeddings"].astype(np.float64)
    feat = np.load(EXPERIMENTS / "rf_cache_test.npz")
    F = feat["features"].astype(np.float64)
    assert F.shape[0] == X.shape[0], (F.shape, X.shape)

    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)

    pca = PCA(n_components=2)
    S = pca.fit_transform(Xn)

    rows = []
    for j, name in enumerate(FEATURE_NAMES):
        y = F[:, j]
        rows.append({
            "feature": name,
            "pearson_pc1": round(float(np.corrcoef(S[:, 0], y)[0, 1]), 4),
            "pearson_pc2": round(float(np.corrcoef(S[:, 1], y)[0, 1]), 4),
            "spearman_pc1": round(float(stats.spearmanr(S[:, 0], y).correlation), 4),
            "spearman_pc2": round(float(stats.spearmanr(S[:, 1], y).correlation), 4),
            "r2_pc1": round(float(_r2(y, S[:, :1])), 4),
            "r2_pc2": round(float(_r2(y, S[:, 1:])), 4),
            "r2_pc1_pc2": round(float(_r2(y, S)), 4),
        })

    out = {
        "n_events": int(len(X)),
        "pca_variance_ratio": [round(float(v), 6) for v in pca.explained_variance_ratio_],
        "correlations": rows,
        "elapsed_s": round(time.time() - t0, 1),
    }
    out_path = EXPERIMENTS / "embedding_axes.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"n_events={out['n_events']:,}  "
          f"pca_variance={out['pca_variance_ratio']}")
    print(f"{'feature':12s} {'r(PC1)':>8s} {'r(PC2)':>8s} "
          f"{'rho(PC1)':>9s} {'rho(PC2)':>9s} {'R2(PC1)':>8s} "
          f"{'R2(PC2)':>8s} {'R2(both)':>8s}")
    for r in rows:
        print(f"{r['feature']:12s} {r['pearson_pc1']:>8.3f} {r['pearson_pc2']:>8.3f} "
              f"{r['spearman_pc1']:>9.3f} {r['spearman_pc2']:>9.3f} "
              f"{r['r2_pc1']:>8.3f} {r['r2_pc2']:>8.3f} {r['r2_pc1_pc2']:>8.3f}")
    print(f"Saved {out_path} ({out['elapsed_s']}s)")


if __name__ == "__main__":
    main()
