"""Probe the dimensionless (Buckingham-pi) feature set.

Answers "why does SR pick Mass/pT and Tau32?" with two independent,
non-symbolic measures on the full 404k test set:

  1. Univariate AUC per feature (labels vs feature alone).
  2. Random-forest feature importances (subsample, fast).

Run: python -m scripts.analysis.compute_feature_probe
Artifact: experiments/physics_sr/probe_features.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from arcefn.utils.paths import EXPERIMENTS

OUT_DIR = EXPERIMENTS / "physics_sr"
NAMES = ["Mass_over_pT", "mSD_over_pT", "Mult", "nSD",
         "sqrt_d12", "sqrt_d23", "Tau21", "Tau32", "zg", "theta_g"]
RF_N = 100_000
RF_EST = 200
RF_DEPTH = 8
SEED = 42


def main() -> None:
    t0 = time.time()
    feats = np.load(EXPERIMENTS / "rf_cache_test.npz")["features"].astype(np.float64)
    logits = np.load(OUT_DIR / "logits.npz")
    labels = logits["labels"].astype(np.int64)
    assert len(feats) == len(labels) == 404000

    pt = np.maximum(logits["jet_pt"].astype(np.float64), 1e-3)
    mass = np.maximum(feats[:, 0], 0.0)
    m_sd = np.maximum(feats[:, 1], 0.0)
    X = np.column_stack([mass / pt, m_sd / pt, feats[:, 2], feats[:, 3],
                         feats[:, 4], feats[:, 5], feats[:, 6], feats[:, 7],
                         feats[:, 8], feats[:, 9]])

    univ = []
    for i, name in enumerate(NAMES):
        v = X[:, i]
        ok = np.isfinite(v)
        auc = float(roc_auc_score(labels[ok], v[ok])) if ok.sum() > 100 else None
        univ.append({"feature": name, "auc": auc,
                     "mean": float(v.mean()), "std": float(v.std())})

    rng = np.random.default_rng(SEED)
    sel = np.sort(rng.choice(len(X), RF_N, replace=False))
    rf = RandomForestClassifier(n_estimators=RF_EST, max_depth=RF_DEPTH,
                                random_state=SEED, n_jobs=4)
    rf.fit(X[sel], labels[sel])
    imp = {name: float(v) for name, v in zip(NAMES, rf.feature_importances_)}
    rf_auc = float(roc_auc_score(labels[sel], rf.predict_proba(X[sel])[:, 1]))

    out = {
        "n_features": len(NAMES),
        "n_events": int(len(X)),
        "rf_subsample": RF_N,
        "univariate_auc": univ,
        "rf_importances": imp,
        "rf_auc_in_sample": rf_auc,
        "elapsed_seconds": time.time() - t0,
    }
    (OUT_DIR / "probe_features.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
