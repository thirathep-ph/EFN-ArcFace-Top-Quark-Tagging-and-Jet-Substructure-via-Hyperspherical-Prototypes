"""
Compute correct mass-window null baselines (B2 fix).

The paper previously compared mass-window AUC to 0.5 ("random guessing").
That null is wrong: a mass-threshold classifier inside a window containing
its threshold is NOT constant, and a majority-class (constant) classifier
already achieves the window prevalence.

Correct nulls, computed here on the same 50k-event subset as
poc_mass_decoupled_accuracy.py:
  1. prevalence       = top fraction in the window
  2. majority accuracy = max(prevalence, 1 - prevalence)   (constant classifier)
  3. mass-only AUC     = AUC of a ranker using jet mass alone, in the same window
  4. mass-only accuracy = best constant/mass-threshold accuracy in the window

Output: experiments/mass_window_null.json
"""
import os
import sys
import time
import json
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arcefn.utils.paths import EXPERIMENTS, DATA_DIR
from arcefn.utils.device import get_device
from arcefn.data.loader import load_awkward, JetTaggingDataset
from arcefn.models.efn_arcface import TopTaggingModel
from arcefn.utils.physics import compute_features

warnings.filterwarnings("ignore")

device, backend = get_device()

model = TopTaggingModel(s=16.0, m=0.5, embedding_dim=64, particle_dim=128).to(device)
sd = torch.load(EXPERIMENTS / "robustness_s16_m05" / "checkpoint.pt", map_location=device, weights_only=False)
model.load_state_dict(sd, strict=True)
model.eval()

print("Loading 50K events ...")
events, labels, weights = load_awkward(DATA_DIR / "test.h5", max_events=50000, lazy=False)
ds = JetTaggingDataset(events, labels, weights)
loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)

all_margins, all_labels = [], []
with torch.no_grad():
    for x, y, w, m, _ in loader:
        x, m = x.to(device), m.to(device)
        _, sim, _ = model(x, mask=m)
        margin = sim[:, 1] - sim[:, 0]
        all_margins.append(margin.cpu().numpy())
        all_labels.append(y.numpy())
all_margins = np.concatenate(all_margins)
all_labels = np.concatenate(all_labels)

print("Computing jet mass via FastJet ...")
t0 = time.time()
jet_mass = compute_features(events)[:, 0]
print(f"  FastJet done in {time.time()-t0:.0f}s")

windows = [(140, 170), (150, 200), (100, 200)]
results = {"n_events": int(len(all_labels)), "windows": []}

for lo, hi in windows:
    inw = (jet_mass >= lo) & (jet_mass < hi)
    y = all_labels[inw]
    s_model = all_margins[inw]
    m_mass = jet_mass[inw]
    n = int(inw.sum())
    n_top = int((y == 1).sum())
    prevalence = n_top / n

    majority_acc = max(prevalence, 1.0 - prevalence)

    # mass-only ranker AUC (mass as the score)
    mass_auc = roc_auc_score(y, m_mass) if (y == 1).any() and (y == 0).any() else 0.5

    # mass-only best-threshold accuracy (scan thresholds at mass grid)
    best_mass_acc = 0.0
    if n > 0:
        for thr in np.percentile(m_mass, np.linspace(0, 100, 1001)):
            pred = (m_mass >= thr).astype(float)
            best_mass_acc = max(best_mass_acc, float((pred == y).mean()))
        best_mass_acc = max(best_mass_acc, majority_acc)

    # model metrics in window (recompute for consistency)
    model_auc = roc_auc_score(y, s_model) if (y == 1).any() and (y == 0).any() else 0.5
    model_acc = float(((s_model > 0).astype(float) == y).mean())

    results["windows"].append({
        "lo": lo, "hi": hi, "n": n, "n_top": n_top,
        "prevalence": prevalence, "majority_acc": majority_acc,
        "mass_only_auc": mass_auc, "mass_only_best_acc": best_mass_acc,
        "model_auc": model_auc, "model_acc": model_acc,
        "auc_gain_over_mass": model_auc - mass_auc,
        "acc_gain_over_majority": model_acc - majority_acc,
    })
    print(f"[{lo},{hi}) n={n} n_top={n_top} prevalence={prevalence:.4f}")
    print(f"   majority_acc={majority_acc:.4f} mass_auc={mass_auc:.4f} mass_best_acc={best_mass_acc:.4f}")
    print(f"   model_auc={model_auc:.4f} (gain {model_auc-mass_auc:+.4f})  model_acc={model_acc:.4f} (gain {model_acc-majority_acc:+.4f})")

out = EXPERIMENTS / "mass_window_null.json"
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out}")
