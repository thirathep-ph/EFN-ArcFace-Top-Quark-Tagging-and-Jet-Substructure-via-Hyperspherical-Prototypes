"""Angular concentration of discovered subclasses on the ArcFace sphere.

For each per-class spectral-clustering partition (k=2, current paper pipeline)
and the spherical-K-means k=2 partition, compute per-subclass directional
statistics on the L2-normalized embeddings:

  - mean resultant length rbar = ||mean(u_i)||  (0 = uniform, 1 = point mass)
  - von Mises-Fisher concentration kappa hat (bounded, MLE root of A_d(kappa)=rbar)
  - effective angular width theta_eff = sqrt(2/kappa) in degrees

Interpretation: kappa is the "angular tightness" a vMF model would assign to
each subclass; a bounded fit avoids the Bessel-overflow degeneracy seen in
unconstrained EM fits on the ~2D submanifold (see subclass_clustering_comparison).

Run: python -m scripts.analysis.compute_subclass_concentration
"""
from __future__ import annotations

import json
import time
from collections import Counter

import numpy as np
from scipy.optimize import brentq
from scipy.special import iv
from sklearn.cluster import KMeans

from arcefn.utils.clustering import spectral_clustering_subclass
from arcefn.utils.paths import EXPERIMENTS
from arcefn.utils.pred_labels import arcface_preds_from_centers, check_cm

EMBEDDINGS = EXPERIMENTS / "robustness_s16_m05" / "embeddings.npz"
CHECKPOINT = EXPERIMENTS / "robustness_s16_m05" / "checkpoint.pt"
OUT = EXPERIMENTS / "subclass_concentration_results.json"
KAPPA_CAP = 600.0


def normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-10)


def vmf_a_d(kappa: float, d: int) -> float:
    nu = d / 2 - 1
    return float(iv(nu, kappa) / iv(nu - 1, kappa) if kappa > 0 else 0.0)


def solve_kappa(rbar: float, d: int) -> float:
    """Bounded MLE: kappa solves A_d(kappa) = rbar (cap guards Bessel overflow)."""
    if rbar < 1e-6:
        return 1e-3
    if rbar >= 1.0:
        return float(KAPPA_CAP)
    try:
        return float(brentq(lambda kp: vmf_a_d(kp, d) - rbar, 1e-4, KAPPA_CAP))
    except ValueError:
        return float(KAPPA_CAP)


def subclass_stats(x: np.ndarray, labels: np.ndarray, d: int) -> dict:
    """Per-subclass (rbar, kappa, theta_eff_deg) with Core = larger cluster."""
    counts = Counter(labels)
    order = sorted(counts.keys(), key=lambda c: -counts[c])
    out = {}
    for idx, c in enumerate(order[:2]):
        sub = x[labels == c]
        rbar = float(np.linalg.norm(sub.mean(axis=0)))
        kappa = solve_kappa(rbar, d)
        # kappa MLE diverges on the ~2D submanifold (rbar->1); the honest
        # scale is the angular spread of the subclass around its mean.
        theta_mean = float(np.degrees(np.arccos(np.clip(rbar, -1.0, 1.0))))
        theta_rms = float(np.degrees(np.sqrt(2.0 * (1.0 - rbar))))
        out[f"subclass_{idx}"] = {
            "n": int(counts[c]),
            "rbar": rbar,
            "kappa_hat": kappa,
            "theta_mean_deg": theta_mean,
            "angular_rms_deg": theta_rms,
        }
    out["kappa_cap_hit"] = any(
        v["kappa_hat"] >= KAPPA_CAP for v in out.values()
    )
    return out


def main() -> None:
    t0 = time.time()
    emb_np = np.load(EMBEDDINGS)
    embeddings, labels = emb_np["embeddings"], emb_np["labels"]
    assert len(embeddings) == len(labels) == 404000

    preds = arcface_preds_from_centers(embeddings, CHECKPOINT)
    check_cm(preds, labels, "concentration")

    report: dict = {
        "n_events": int(len(embeddings)),
        "embedding_dim": int(embeddings.shape[1]),
        "kappa_cap": KAPPA_CAP,
        "partition": "predicted_labels_primary",
        "classes": {},
    }
    for cls in [0, 1]:
        mask = preds == cls
        x = normalize(embeddings[mask])
        cls_name = "QCD" if cls == 0 else "Top"
        entry: dict = {"n": int(mask.sum()), "partitions": {}}

        sc_labels, k_sc = spectral_clustering_subclass(x, max_k=5)
        entry["partitions"]["spectral_eigengap"] = {
            "k": k_sc,
            "subclasses": subclass_stats(x, sc_labels, x.shape[1]),
        }

        km = KMeans(n_clusters=2, random_state=42, n_init=10).fit(x)
        entry["partitions"]["spherical_kmeans_k2"] = {
            "subclasses": subclass_stats(x, km.labels_, x.shape[1]),
        }

        report["classes"][cls_name] = entry
        print(f"\n[{cls_name}] n={entry['n']:,}  (t={time.time()-t0:.0f}s)")
        print(json.dumps(entry, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))

    with open(OUT, "w") as f:
        json.dump(report, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"\nSaved: {OUT}\n[COMPLETE] ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
