"""Mass-binned DGLAP control analysis.

Question (reviewer M7): are the QCD/Top subclass differences in the zg
power-law exponent alpha merely an echo of jet mass (the subclasses differ
in mass), or do they reflect genuine splitting-structure differences?

Method: hold mass fixed by binning each class into mass quantiles, then fit
the AP power law P(zg)=A zg^alpha (1-zg)^beta separately for the Core and
Edge subclass within each bin. If alpha_Core - alpha_Edge persists within
bins, the subclass structure is not a mass echo. Reuses the cached
embeddings (subclass discovery) and feature matrix (mass, zg) — no forward
pass and no FastJet recomputation.

Run: python -m scripts.analysis.compute_mass_matched_dglap
"""
from __future__ import annotations

import json
import time
from collections import Counter

import numpy as np

from arcefn.utils.clustering import spectral_clustering_subclass
from arcefn.utils.paths import EXPERIMENTS
from scripts.analysis.compute_dglap_fit import SUBCLASS_NAMES, fit_zg

EMBEDDINGS = EXPERIMENTS / "robustness_s16_m05" / "embeddings.npz"
FEATURES = EXPERIMENTS / "rf_cache_test.npz"
OUT = EXPERIMENTS / "mass_matched_dglap.json"
N_MASS_BINS = 5
MIN_SUBCLASS_N = 2000


def discover_subclasses(embeddings: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Primary PREDICTED-label partition (what the model sees, max_k=5)."""
    from arcefn.utils.pred_labels import arcface_preds_from_centers, check_cm
    preds = arcface_preds_from_centers(
        embeddings, EXPERIMENTS / "robustness_s16_m05" / "checkpoint.pt")
    check_cm(preds, labels, "mass_matched")
    subclass_labels = np.full(len(labels), -1)
    for cls in [0, 1]:
        mask = preds == cls
        sc_labels, n_clusters = spectral_clustering_subclass(embeddings[mask], max_k=5)
        if n_clusters >= 2:
            counts = Counter(sc_labels)
            order = sorted(counts.keys(), key=lambda k: -counts[k])
            label_map = {order[0]: 0, order[1]: 1}
            sc_labels = np.array([label_map.get(l, 0) for l in sc_labels])
        sc_labels = sc_labels % 2
        subclass_labels[mask] = sc_labels + (2 * cls)
    return subclass_labels


def main() -> None:
    t0 = time.time()
    emb_np = np.load(EMBEDDINGS)
    embeddings, labels = emb_np["embeddings"], emb_np["labels"]
    feat = np.load(FEATURES)
    mass_all = feat["features"][:, 0].astype(np.float64)
    zg_all = feat["features"][:, 8].astype(np.float64)
    assert len(embeddings) == len(labels) == len(mass_all) == len(zg_all) == 404000

    subclass_labels = discover_subclasses(embeddings, labels)

    report: dict = {"n_events": int(len(embeddings)), "n_mass_bins": N_MASS_BINS,
                    "min_subclass_n": MIN_SUBCLASS_N,
                    "partition": "predicted_labels_primary", "classes": {}}

    from arcefn.utils.pred_labels import arcface_preds_from_centers
    preds = arcface_preds_from_centers(
        embeddings, EXPERIMENTS / "robustness_s16_m05" / "checkpoint.pt")
    for cls, cls_name in [(0, "QCD"), (1, "Top")]:
        cmask = preds == cls
        mass = mass_all[cmask]
        zg = zg_all[cmask]
        sub = subclass_labels[cmask]
        entry: dict = {"n": int(cmask.sum()), "overall": {}, "mass_bins": []}

        for sc in [0, 1]:
            sm = sub == (2 * cls + sc)
            fit = fit_zg(zg[sm])
            if fit:
                entry["overall"][SUBCLASS_NAMES[2 * cls + sc]] = {
                    "alpha": fit["alpha"], "beta": fit["beta"],
                    "r2": fit["r2"], "n": fit["n"],
                    "mean_mass": float(mass[sm].mean()),
                    "mean_tau32": float(feat["features"][cmask][:, 7][sm].mean())}
        a_core = entry["overall"][SUBCLASS_NAMES[2 * cls]]["alpha"]
        a_edge = entry["overall"][SUBCLASS_NAMES[2 * cls + 1]]["alpha"]
        entry["overall_delta_alpha"] = a_core - a_edge

        edges = np.quantile(mass, np.linspace(0.0, 1.0, N_MASS_BINS + 1))
        edges[0], edges[-1] = -np.inf, np.inf
        deltas = []
        weights = []
        for b in range(N_MASS_BINS):
            bmask = (mass >= edges[b]) & (mass < edges[b + 1])
            row: dict = {"lo": float(edges[b]), "hi": float(edges[b + 1]),
                         "n_total": int(bmask.sum())}
            fits = {}
            ok = True
            for sc in [0, 1]:
                sel = bmask & (sub == (2 * cls + sc))
                row[f"n_{sc}"] = int(sel.sum())
                fit = fit_zg(zg[sel]) if sel.sum() >= MIN_SUBCLASS_N else None
                if fit:
                    fits[sc] = fit
                    row[SUBCLASS_NAMES[2 * cls + sc]] = {
                        "alpha": fit["alpha"], "beta": fit["beta"],
                        "r2": fit["r2"], "n": fit["n"]}
                else:
                    ok = False
            if ok:
                row["delta_alpha"] = fits[0]["alpha"] - fits[1]["alpha"]
                deltas.append(row["delta_alpha"])
                weights.append(min(row["n_0"], row["n_1"]))
            entry["mass_bins"].append(row)

        if deltas:
            entry["mass_matched_delta_alpha"] = float(
                np.average(deltas, weights=weights))
        report["classes"][cls_name] = entry
        print(f"[{cls_name}] overall dAlpha={entry['overall_delta_alpha']:+.3f}")

    with open(OUT, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"\nSaved: {OUT}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
