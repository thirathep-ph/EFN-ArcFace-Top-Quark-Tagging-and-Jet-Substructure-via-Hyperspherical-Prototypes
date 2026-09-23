"""Subclass clustering on the intrinsic 2D latent (per-class PCA-2D of embeddings).

Motivation: vMF mixture EM fails on the full S^63 because the embeddings live on
an ~2D submanifold (TwoNN 2.17, PR 1.38) — kappa-hat MLE diverges and BIC has
no minimum. Fitting on the 2D latent matches the data's real geometry:
  - GMM k=2 (BIC for k=2..4)     soft memberships
  - vMF-on-S^1 (normalized 2D)   bounded kappa, soft memberships
  - spherical K-means (2D)
  - HDBSCAN (density-based arbiter: continuum -> few clusters, bimodal -> 2)

Metrics: ARI vs spectral SC partition, cosine silhouette (64-d), DGLAP
power-law fits per partition (physics arbiter), soft-membership entropy vs mass.

Run: python -m scripts.analysis.compute_subclass_2dlatent
"""
from __future__ import annotations

import json
import time
from collections import Counter

import numpy as np
from scipy.optimize import brentq
from scipy.special import iv
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture

from arcefn.utils.clustering import spectral_clustering_subclass
from arcefn.utils.paths import EXPERIMENTS
from scripts.analysis.compute_dglap_fit import SUBCLASS_NAMES, fit_zg

EMBEDDINGS = EXPERIMENTS / "robustness_s16_m05" / "embeddings.npz"
FEATURES = EXPERIMENTS / "rf_cache_test.npz"
OUT = EXPERIMENTS / "subclass_2dlatent_results.json"
SILHOUETTE_SAMPLE = 30000
SEED = 42


def normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-10)


def vmf_a_d(kappa: float, d: int) -> float:
    nu = d / 2 - 1
    return float(iv(nu, kappa) / iv(nu - 1, kappa) if kappa > 0 else 0.0)


def vmf_log_c(kappa: float | np.ndarray, d: int) -> float | np.ndarray:
    k = np.asarray(kappa, dtype=float)
    nu = d / 2 - 1
    return (d / 2 - 1) * np.log(k) - (d / 2) * np.log(2 * np.pi) - np.log(np.maximum(iv(nu, k), 1e-300))


def _solve_kappa(rbar: float, d: int, lo: float = 1e-4, hi: float = 600.0) -> float:
    if rbar < 1e-6:
        return 1e-3
    try:
        return float(brentq(lambda kp: vmf_a_d(kp, d) - rbar, lo, hi))
    except ValueError:
        return float(hi)


def vmf_em_s1(x2: np.ndarray, k: int = 2, max_iter: int = 100, tol: float = 1e-6) -> tuple:
    """Soft vMF mixture EM on the unit circle (d=2): bounded kappa guaranteed."""
    x = normalize(x2)
    rng = np.random.default_rng(SEED)
    init = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(x)
    mus = normalize(init.cluster_centers_)
    kappas = np.full(k, 5.0)
    pis = np.full(k, 1.0 / k)
    ll = -np.inf
    for _ in range(max_iter):
        cos = x @ mus.T
        log_r = np.log(pis + 1e-300) + kappas[None, :] * cos + vmf_log_c(kappas, 2)[None, :]
        log_r -= log_r.max(axis=1, keepdims=True)
        r = np.exp(log_r)
        r /= r.sum(axis=1, keepdims=True)
        n_k = r.sum(axis=0) + 1e-12
        new_mus = normalize(r.T @ x)
        rbar = np.linalg.norm(r.T @ x, axis=1) / n_k
        new_kappas = np.array([_solve_kappa(rb, 2) for rb in rbar])
        new_pis = n_k / n_k.sum()
        terms = np.log(new_pis + 1e-300) + new_kappas[None, :] * cos + vmf_log_c(new_kappas, 2)[None, :]
        ll_new = float(np.logaddexp.reduce(terms, axis=1).sum())
        mus, kappas, pis = new_mus, new_kappas, new_pis
        if abs(ll_new - ll) < tol * abs(ll_new):
            break
        ll = ll_new
    resp = np.exp(np.log(pis + 1e-300) + kappas[None, :] * (x @ mus.T) + vmf_log_c(kappas, 2)[None, :])
    resp /= resp.sum(axis=1, keepdims=True)
    return resp, mus, kappas, pis, ll


def fit_partition(zg_cls: np.ndarray, sub_labels: np.ndarray, cls: int) -> dict:
    counts = Counter(sub_labels)
    order = sorted(counts.keys(), key=lambda c: -counts[c])
    out = {}
    for idx, c in enumerate(order[:2]):
        fit = fit_zg(zg_cls[sub_labels == c])
        if fit:
            name = SUBCLASS_NAMES[2 * cls + idx]
            out[name] = {"alpha": fit["alpha"], "beta": fit["beta"],
                         "r2": fit["r2"], "n": fit["n"]}
    return out


def _silhouette(x64: np.ndarray, lab: np.ndarray) -> float:
    n = min(SILHOUETTE_SAMPLE, len(x64))
    if len(np.unique(lab)) < 2:
        return float("nan")
    return float(silhouette_score(x64, lab, metric="cosine", sample_size=n, random_state=SEED))


def entropy_of(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def main() -> None:
    t0 = time.time()
    emb_np = np.load(EMBEDDINGS)
    embeddings, labels = emb_np["embeddings"], emb_np["labels"]
    feat = np.load(FEATURES)
    features, zg_all, mass_all = feat["features"], feat["features"][:, 8], feat["features"][:, 0]
    assert len(embeddings) == len(labels) == 404000

    report: dict = {"n_events": len(embeddings), "latent_dim": 2, "classes": {}}
    for cls in [0, 1]:
        mask = labels == cls
        x64 = normalize(embeddings[mask])
        zg_cls = zg_all[mask]
        mass_cls = mass_all[mask]
        cls_name = "QCD" if cls == 0 else "Top"
        entry: dict = {"n": int(mask.sum()), "partitions": {}}

        pca = PCA(n_components=2, random_state=SEED).fit(x64)
        z2 = pca.transform(x64)  # (n,2)
        explained = pca.explained_variance_ratio_.tolist()

        sc_labels, k_sc = spectral_clustering_subclass(x64, max_k=5)
        counts = Counter(sc_labels)
        order = sorted(counts.keys(), key=lambda c: -counts[c])
        sc_mapped = np.array([order.index(l) if l in order else 0 for l in sc_labels])

        # --- GMM k=2..4 (BIC) + soft membership (k=2) ---
        gmm_entry = {}
        gmm2 = None
        for k in (2, 3, 4):
            g = GaussianMixture(n_components=k, covariance_type="full", random_state=SEED).fit(z2)
            gmm_entry[f"k{k}"] = {"bic": float(g.bic(z2)), "aic": float(g.aic(z2))}
            if k == 2:
                gmm2 = g
        assert gmm2 is not None
        resp = gmm2.predict_proba(z2)
        gmm_labels = gmm2.predict(z2)
        counts = Counter(gmm_labels)
        order = sorted(counts.keys(), key=lambda c: -counts[c])
        gmm_mapped = np.array([order.index(l) if l in order else 0 for l in gmm_labels])
        gmm_entry["k2_final"] = {
            "n0": int(counts[order[0]]), "n1": int(counts[order[1]]),
            "ari_vs_spectral": float(adjusted_rand_score(sc_mapped, gmm_mapped)),
            "silhouette_64d": _silhouette(x64, gmm_mapped),
            "dglap": fit_partition(zg_cls, gmm_mapped, cls),
        }
        entry["partitions"]["gmm_2d"] = gmm_entry

        # --- vMF on S^1 (normalized 2D) ---
        resp_v, mus_v, kappas_v, pis_v, ll_v = vmf_em_s1(z2)
        v_labels = np.argmax(resp_v, axis=1)
        counts = Counter(v_labels)
        order = sorted(counts.keys(), key=lambda c: -counts[c])
        v_mapped = np.array([order.index(l) if l in order else 0 for l in v_labels])
        entry["partitions"]["vmf_s1"] = {
            "n0": int(counts[order[0]]), "n1": int(counts[order[1]]),
            "kappas": [float(v) for v in kappas_v],
            "log_likelihood": ll_v,
            "ari_vs_spectral": float(adjusted_rand_score(sc_mapped, v_mapped)),
            "silhouette_64d": _silhouette(x64, v_mapped),
            "dglap": fit_partition(zg_cls, v_mapped, cls),
        }

        # --- spherical K-means on 2D ---
        km = KMeans(n_clusters=2, random_state=SEED, n_init=10).fit(normalize(z2))
        km_labels = km.labels_
        counts = Counter(km_labels)
        order = sorted(counts.keys(), key=lambda c: -counts[c])
        km_mapped = np.array([order.index(l) if l in order else 0 for l in km_labels])
        entry["partitions"]["skm_2d"] = {
            "n0": int(counts[order[0]]), "n1": int(counts[order[1]]),
            "ari_vs_spectral": float(adjusted_rand_score(sc_mapped, km_mapped)),
            "silhouette_64d": _silhouette(x64, km_mapped),
            "dglap": fit_partition(zg_cls, km_mapped, cls),
        }

        # --- HDBSCAN (density-based arbiter) ---
        hd = HDBSCAN(min_cluster_size=20000, min_samples=10, metric="euclidean").fit(z2)
        hd_labels = hd.labels_
        n_clusters_hd = len(set(hd_labels)) - (1 if -1 in hd_labels else 0)
        noise_frac = float((hd_labels == -1).mean())
        hd_valid = hd_labels[hd_labels != -1]
        hd_ari = float(adjusted_rand_score(sc_mapped[hd_labels != -1], hd_valid)) if len(hd_valid) > 0 else None
        entry["partitions"]["hdbscan_2d"] = {
            "n_clusters": n_clusters_hd, "noise_fraction": noise_frac,
            "ari_vs_spectral_on_core": hd_ari,
        }

        # --- soft-membership entropy vs mass (option B) ---
        ent_gmm = entropy_of(resp)
        ent_vmf = entropy_of(resp_v)
        order2 = sorted(np.unique(gmm_labels), key=lambda c: -int((gmm_labels == c).sum()))
        for name, ent, lab in [("gmm", ent_gmm, gmm_labels), ("vmf", ent_vmf, v_labels)]:
            corr = float(np.corrcoef(ent, mass_cls)[0, 1])
            per_sub = {}
            for c in order2:
                per_sub[f"sub{c}"] = {"n": int((lab == c).sum()),
                                      "mean_entropy": float(ent[lab == c].mean()),
                                      "mean_mass": float(mass_cls[lab == c].mean())}
            entry.setdefault("entropy", {})[name] = {
                "corr_entropy_mass": corr,
                "fraction_entropy_gt_0.5": float((ent > 0.5).mean()),
                "per_subclass": per_sub,
            }

        entry["pca_explained_variance"] = explained
        report["classes"][cls_name] = entry
        print(f"\n[{cls_name}] n={entry['n']:,} PCA-2D explained={explained} (t={time.time()-t0:.0f}s)")

    with open(OUT, "w") as f:
        json.dump(report, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"\nSaved: {OUT}\n[COMPLETE] ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
