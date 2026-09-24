"""Compare subclass clustering methods on the ArcFace embedding sphere.

Methods compared per class (predicted labels, primary partition):
  - spectral clustering + eigengap (current analysis pipeline)
  - spherical K-means (KMeans on L2-normalized embeddings), k in 2..5
  - vMF mixture via EM (soft assignments, per-cluster kappa), k in 2..5

Metrics: cosine silhouette (sampled), Davies-Bouldin, vMF BIC, NMI/ARI
agreement between methods, and DGLAP power-law fit R^2 per partition
(physics arbiter, zg from the cached feature matrix).

Run: python -m scripts.analysis.compute_subclass_clustering
"""
from __future__ import annotations

import json
import time
from collections import Counter

import numpy as np
from scipy.optimize import brentq
from scipy.special import iv
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)

from arcefn.utils.clustering import spectral_clustering_subclass
from arcefn.utils.paths import EXPERIMENTS
from scripts.analysis.compute_dglap_fit import SUBCLASS_NAMES, fit_zg

EMBEDDINGS = EXPERIMENTS / "robustness_s16_m05" / "embeddings.npz"
FEATURES = EXPERIMENTS / "rf_cache_test.npz"
OUT = EXPERIMENTS / "subclass_clustering_comparison.json"
SILHOUETTE_SAMPLE = 30000


def normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-10)


def vmf_a_d(kappa: float, d: int) -> float:
    nu = d / 2 - 1
    return float(iv(nu, kappa) / iv(nu - 1, kappa) if kappa > 0 else 0.0)


def vmf_log_c(kappa: float | np.ndarray, d: int) -> float | np.ndarray:
    k = np.asarray(kappa, dtype=float)
    nu = d / 2 - 1
    return (d / 2 - 1) * np.log(k) - (d / 2) * np.log(2 * np.pi) - np.log(np.maximum(iv(nu, k), 1e-300))


def vmf_em(x: np.ndarray, k: int, max_iter: int = 150, tol: float = 1e-6, seed: int = 42) -> tuple:
    """Soft vMF mixture EM. Returns (mus, kappas, pis, log_likelihood)."""
    rng = np.random.default_rng(seed)
    init = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(x)
    mus = normalize(init.cluster_centers_)
    kappas = np.full(k, 20.0)
    pis = np.full(k, 1.0 / k)
    ll = -np.inf
    for _ in range(max_iter):
        cos = x @ mus.T
        log_r = np.log(pis + 1e-300) + kappas[None, :] * cos + vmf_log_c(kappas, x.shape[1])[None, :]
        log_r -= log_r.max(axis=1, keepdims=True)
        r = np.exp(log_r)
        r /= r.sum(axis=1, keepdims=True)
        n_k = r.sum(axis=0) + 1e-12
        new_mus = normalize(r.T @ x)
        rbar = np.linalg.norm(r.T @ x, axis=1) / n_k
        new_kappas = np.array([_solve_kappa(rb, x.shape[1]) for rb in rbar])
        new_pis = n_k / n_k.sum()
        ll_new = _log_likelihood(x, new_mus, new_kappas, new_pis)
        mus, kappas, pis = new_mus, new_kappas, new_pis
        if abs(ll_new - ll) < tol * abs(ll_new):
            ll = ll_new
            break
        ll = ll_new
    return mus, kappas, pis, ll


def _solve_kappa(rbar: float, d: int, lo: float = 1e-4, hi: float = 600.0) -> float:
    if rbar < 1e-6:
        return 1e-3
    try:
        return float(brentq(lambda kp: vmf_a_d(kp, d) - rbar, lo, hi))
    except ValueError:
        return float(hi)


def _log_likelihood(x: np.ndarray, mus: np.ndarray, kappas: np.ndarray, pis: np.ndarray) -> float:
    d = x.shape[1]
    terms = np.log(pis + 1e-300) + kappas[None, :] * (x @ mus.T) + vmf_log_c(kappas, d)[None, :]
    return float(np.logaddexp.reduce(terms, axis=1).sum())


def vmf_bic(ll: float, n: int, d: int, k: int) -> float:
    n_params = (k - 1) + k * d + k
    return -2 * ll + n_params * np.log(n)


def fit_partition(zg_cls: np.ndarray, sub_labels: np.ndarray, cls: int) -> dict:
    """DGLAP power-law fit per subclass (larger cluster mapped to Core)."""
    out = {}
    counts = Counter(sub_labels)
    order = sorted(counts.keys(), key=lambda c: -counts[c])
    for idx, c in enumerate(order[:2]):
        fit = fit_zg(zg_cls[sub_labels == c])
        if fit:
            name = SUBCLASS_NAMES[2 * cls + idx]
            out[name] = {"alpha": fit["alpha"], "beta": fit["beta"],
                         "r2": fit["r2"], "n": fit["n"]}
    return out


def main() -> None:
    t0 = time.time()
    emb_np = np.load(EMBEDDINGS)
    embeddings, labels = emb_np["embeddings"], emb_np["labels"]
    feat = np.load(FEATURES)
    zg_all = feat["features"][:, 8]
    assert len(embeddings) == len(labels) == len(zg_all) == 404000

    from arcefn.utils.pred_labels import arcface_preds_from_centers, check_cm
    preds = arcface_preds_from_centers(
        embeddings, EXPERIMENTS / "robustness_s16_m05" / "checkpoint.pt")
    check_cm(preds, labels, "subclass_clustering")

    report: dict = {"n_events": len(embeddings),
                    "partition": "predicted_labels_primary", "methods": {}}
    for cls in [0, 1]:
        mask = preds == cls
        x = normalize(embeddings[mask])
        zg_cls = zg_all[mask]
        cls_name = "QCD" if cls == 0 else "Top"
        entry: dict = {"n": int(mask.sum()), "partitions": {}}

        sc_labels, k_sc = spectral_clustering_subclass(x, max_k=5)
        counts = Counter(sc_labels)
        order = sorted(counts.keys(), key=lambda c: -counts[c])
        sc_mapped = np.array([order.index(l) if l in order else 0 for l in sc_labels])
        entry["partitions"]["spectral_eigengap"] = {
            "k": k_sc, "n0": int(counts[order[0]]), "n1": int(counts[order[1]]),
            "silhouette": _silhouette(x, sc_mapped),
            "davies_bouldin": float(davies_bouldin_score(x, sc_mapped)),
            "dglap": fit_partition(zg_cls, sc_mapped, cls),
        }

        skm = {}
        vmf = {}
        for k in range(2, 6):
            km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(x)
            km_labels = km.labels_
            skm[f"k{k}"] = {"silhouette": _silhouette(x, km_labels),
                            "davies_bouldin": float(davies_bouldin_score(x, km_labels)),
                            "nmi_vs_spectral": float(normalized_mutual_info_score(sc_mapped, km_labels)),
                            "ari_vs_spectral": float(adjusted_rand_score(sc_mapped, km_labels))}
            mus, kappas, pis, ll = vmf_em(x, k)
            labels_k = np.argmax(np.log(pis + 1e-300) + kappas[None, :] * (x @ mus.T), axis=1)
            vmf[f"k{k}"] = {"log_likelihood": ll, "bic": vmf_bic(ll, len(x), x.shape[1], k),
                            "kappas": [float(v) for v in kappas],
                            "silhouette": _silhouette(x, labels_k),
                            "nmi_vs_spectral": float(normalized_mutual_info_score(sc_mapped, labels_k)),
                            "ari_vs_spectral": float(adjusted_rand_score(sc_mapped, labels_k))}
        entry["partitions"]["spherical_kmeans"] = skm
        entry["partitions"]["vmf_mixture"] = vmf

        for k in (2, 3):
            km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(x)
            km_labels = km.labels_
            counts = Counter(km_labels)
            order = sorted(counts.keys(), key=lambda c: -counts[c])
            km_mapped = np.array([order.index(l) if l in order else 0 for l in km_labels])
            entry["partitions"].setdefault("skm_dglap", {})[f"k{k}"] = fit_partition(zg_cls, km_mapped, cls)
            mus, kappas, pis, ll = vmf_em(x, k)
            v_labels = np.argmax(np.log(pis + 1e-300) + kappas[None, :] * (x @ mus.T), axis=1)
            counts = Counter(v_labels)
            order = sorted(counts.keys(), key=lambda c: -counts[c])
            v_mapped = np.array([order.index(l) if l in order else 0 for l in v_labels])
            entry["partitions"].setdefault("vmf_dglap", {})[f"k{k}"] = fit_partition(zg_cls, v_mapped, cls)

        report["methods"][cls_name] = entry
        print(f"\n[{cls_name}] n={entry['n']:,}  (t={time.time()-t0:.0f}s)")

    with open(OUT, "w") as f:
        json.dump(report, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"\nSaved: {OUT}\n[COMPLETE] ({time.time()-t0:.0f}s)")


def _silhouette(x: np.ndarray, lab: np.ndarray) -> float:
    n = min(SILHOUETTE_SAMPLE, len(x))
    if len(np.unique(lab)) < 2:
        return float("nan")
    return float(silhouette_score(x, lab, metric="cosine", sample_size=n, random_state=42))


if __name__ == "__main__":
    main()
