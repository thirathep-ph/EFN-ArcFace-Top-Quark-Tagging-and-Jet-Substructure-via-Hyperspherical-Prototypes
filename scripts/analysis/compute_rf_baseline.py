"""Random Forest baseline on hand-crafted substructure features.

Trains a RandomForestClassifier on the 10 engineered physics features
(Mass, mSD, Mult, nSD, sqrt(d12), sqrt(d23), tau21, tau32, zg, theta_g)
extracted by ``arcefn.utils.physics.compute_features`` and reports the
same metrics as Table 1 (accuracy, ROC AUC, Rejection@50%TPR) using the
identical methodology as ``compute_bootstrap_ci.py`` so the numbers are
directly comparable to the EFN baselines.

The RF does not see the raw per-constituent 4-vectors; it operates on
hand-crafted, jet-clustering-dependent features.  It is reported as an
additional baseline to show what a classical method achieves given the
same physical observables.

For a fair comparison the RF is trained on the held-out validation split
(``val.h5``) and evaluated on the full test split (``test.h5``), mirroring
how the EFN models are trained on the training split and evaluated on the
test split.

Usage
-----
Run from the repo root so that ``arcefn`` is importable::

    .\\.venv\\Scripts\\python.exe -m scripts.analysis.compute_rf_baseline \\
        --max-train-events 403000 --max-events 404000 --n-estimators 300

Output
------
``experiments/rf_baseline_results.json``
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, roc_curve, auc

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from arcefn.utils.paths import EXPERIMENTS, DATA_DIR
from arcefn.data.loader import load_awkward


def load_features(split: str, max_events: int, seed: int = 42):
    """Load events for a split, compute the 10 features, cache to .npz."""
    cache = EXPERIMENTS / f'rf_cache_{split}.npz'
    if cache.exists():
        data = np.load(cache)
        return data['features'], data['labels']
    rng = np.random.default_rng(seed)
    events, labels, _ = load_awkward(str(DATA_DIR / f'{split}.h5'),
                                     max_events=max_events, lazy=True)
    total = len(events)
    n_use = min(max_events, total)
    if n_use < total:
        idx = np.sort(rng.choice(total, n_use, replace=False))
    else:
        idx = np.arange(total)
    events = events[idx]
    labels = labels[idx]
    print(f"Computing features for {split}.h5 ({n_use} events)...")
    from arcefn.utils.physics import compute_features
    features = compute_features(events)
    np.savez(cache, features=features, labels=labels)
    print(f"Cached: {cache}")
    return features, labels


def compute_rejection_at_tpr(labels, scores, target_tprs=(0.3, 0.5, 0.7, 0.8, 0.9, 0.95)):
    """Rejection (1/FPR) at fixed signal TPR, matching the reported method."""
    fpr, tpr, _ = roc_curve(labels, scores)
    rates = {}
    for target in target_tprs:
        idx = np.argmin(np.abs(tpr - target))
        rates[f"tpr_{int(target*100)}"] = {
            "tpr": float(tpr[idx]),
            "fpr": float(fpr[idx]),
            "rejection": float(1.0 / fpr[idx]) if fpr[idx] > 0 else None,
        }
    return rates


def main() -> None:
    parser = argparse.ArgumentParser(description="Random Forest baseline")
    parser.add_argument("--max-train-events", type=int, default=403000,
                        help="Number of train events from val.h5 (seeded random subsample)")
    parser.add_argument("--max-events", type=int, default=404000,
                        help="Number of eval events from test.h5 (seeded random subsample)")
    parser.add_argument("--n-estimators", type=int, default=300,
                        help="Number of trees")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-depth", type=int, default=None,
                        help="Optional tree depth cap (None = unlimited)")
    args = parser.parse_args()

    print("=" * 60)
    print("Random Forest baseline")
    print("=" * 60)

    features_tr, y_train = load_features('val', args.max_train_events, args.seed)
    features_te, y_test = load_features('test', args.max_events, args.seed)
    print(f"Train: {features_tr.shape} from val.h5 (seed {args.seed})")
    print(f"Test:  {features_te.shape} from test.h5 (seed {args.seed})")
    print(f"Class balance train: {np.bincount(y_train)} / test: {np.bincount(y_test)}")

    print(f"Training RandomForest (n_estimators={args.n_estimators}, "
          f"max_depth={args.max_depth})...")
    t0 = time.time()
    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.seed,
        n_jobs=-1,
    )
    clf.fit(features_tr, y_train)
    t_train = time.time() - t0
    print(f"Trained in {t_train:.1f}s")

    acc = clf.score(features_te, y_test)
    probs = clf.predict_proba(features_te)[:, 1]
    auc_val = roc_auc_score(y_test, probs)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, clf.predict(features_te),
                                                               average="binary")
    rej = compute_rejection_at_tpr(y_test, probs)

    names = ["Mass", "mSD", "Mult", "nSD", "sqrt(d12)", "sqrt(d23)",
             "tau21", "tau32", "zg", "theta_g"]
    importances = dict(zip(names, clf.feature_importances_.tolist()))

    results = {
        "model": "RandomForest",
        "train_split": "val.h5",
        "test_split": "test.h5",
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "seed": args.seed,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "features": names,
        "train_time_seconds": round(t_train, 2),
        "accuracy": round(float(acc), 6),
        "roc_auc": round(float(auc_val), 6),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
        "rejection": {k: round(v["rejection"], 3) if v["rejection"] else None
                      for k, v in rej.items()},
        "rejection_at_50pct_tpr": round(float(rej["tpr_50"]["rejection"]), 3),
        "feature_importance": importances,
    }

    out_path = EXPERIMENTS / 'rf_baseline_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()