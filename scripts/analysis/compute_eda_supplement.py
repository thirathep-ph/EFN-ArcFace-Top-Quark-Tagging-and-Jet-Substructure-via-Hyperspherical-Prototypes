"""EDA supplement: sigma(pT), sigma(M), per-class means, correlations.

Computes population-level descriptive statistics not stored by
compute_eda.py, on a seeded random 100k-event subsample of test.h5:
  - jet pT std (from constituent sums, E-scheme) and mass std
  - per-class tau32 / multiplicity means
  - correlation Mass-mSD and sqrt(d12)-sqrt(d23)
Saves experiments/eda_supplement_results.json.
"""
import json
import sys
from pathlib import Path

import awkward as ak
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from arcefn.utils.paths import EXPERIMENTS, DATA_DIR
from arcefn.data.loader import load_awkward
from arcefn.utils.physics import compute_features

rng = np.random.default_rng(42)
events, labels, _ = load_awkward(str(DATA_DIR / 'test.h5'), max_events=100000, lazy=True)
total = len(events)
idx = np.sort(rng.choice(total, 100000, replace=False))
events = events[idx]
labels = labels[idx]

features = compute_features(events)
mass = features[:, 0]
t32 = features[:, 7]
mult = features[:, 2]
d12, d23 = features[:, 4], features[:, 5]

px = np.asarray(ak.sum(events.px, axis=1))
py = np.asarray(ak.sum(events.py, axis=1))
pt = np.sqrt(px**2 + py**2)

qcd = labels == 0
top = labels == 1

corr = np.corrcoef(np.stack([mass, features[:, 1]]), rowvar=True)[0, 1]
corr_d = np.corrcoef(np.stack([d12, d23]), rowvar=True)[0, 1]

results = {
    "n_events": int(len(labels)),
    "seed": 42,
    "pt": {"mean": float(pt.mean()), "std": float(pt.std(ddof=1))},
    "mass": {"mean": float(mass.mean()), "std": float(mass.std(ddof=1))},
    "tau32": {
        "top_mean": float(t32[top].mean()),
        "qcd_mean": float(t32[qcd].mean()),
    },
    "multiplicity": {
        "top_mean": float(mult[top].mean()),
        "qcd_mean": float(mult[qcd].mean()),
    },
    "corr_mass_msd": float(corr),
    "corr_d12_d23": float(corr_d),
}

out_path = EXPERIMENTS / 'eda_supplement_results.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))
print(f"Saved: {out_path}")
