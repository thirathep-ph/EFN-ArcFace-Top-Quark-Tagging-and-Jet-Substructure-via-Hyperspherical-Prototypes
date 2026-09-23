"""Cohen's d on the PREDICTED partition (traceable source for paper).

Uses canonical model predictions + spectral subclasses per predicted class
(same pipeline as compute_mass_matched_dglap.py), mass/tau32 from rf_cache.
Saves experiments/predicted_effect_sizes.json
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from arcefn.utils.paths import DATA_DIR, EXPERIMENTS
from arcefn.utils.device import get_device
from arcefn.utils.model_loading import load_model_from_dir
from arcefn.data.loader import load_awkward, JetTaggingDataset
from arcefn.utils.clustering import spectral_clustering_subclass

MODEL_DIR = EXPERIMENTS / "robustness_s16_m05"
BATCH_SIZE = 2048


def cohen(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1, n2 = len(a), len(b)
    s = np.sqrt(((n1 - 1) * a.var() + (n2 - 1) * b.var()) / (n1 + n2 - 2))
    return float((a.mean() - b.mean()) / s)


def main():
    device, _ = get_device()
    model = load_model_from_dir(MODEL_DIR, device)
    model.eval()
    events, labels, weights = load_awkward(str(DATA_DIR / 'test.h5'), lazy=False)
    loader = DataLoader(JetTaggingDataset(events, labels, weights),
                        batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    all_emb, all_pred = [], []
    with torch.no_grad():
        for x, y, w, m, _ in loader:
            x, m = x.to(device), m.to(device)
            out = model(x, mask=m)
            logits, emb = out[0], out[-1]
            all_emb.append(F.normalize(emb, p=2, dim=1).cpu().numpy())
            all_pred.append(torch.argmax(logits, dim=1).cpu().numpy())
    emb = np.concatenate(all_emb)
    pred = np.concatenate(all_pred)
    feat = np.load(EXPERIMENTS / "rf_cache_test.npz")["features"]
    mass, tau = feat[:, 0], feat[:, 7]
    out = {"n_events": int(len(pred)), "partition": "predicted_labels_primary", "subclasses": {}}
    for cls, cname in [(0, "QCD"), (1, "Top")]:
        X = emb[pred == cls]
        sc, k = spectral_clustering_subclass(X, max_k=5)
        # size convention: larger cluster = Core (0)
        counts = np.bincount(sc)
        core_id = int(np.argmax(counts))
        is_core = sc == core_id
        idx = np.where(pred == cls)[0]
        core_idx, edge_idx = idx[is_core], idx[~is_core]
        res = {"k": int(k), "n_core": int(is_core.sum()), "n_edge": int((~is_core).sum())}
        res["mass_mean_core"] = float(mass[core_idx].mean())
        res["mass_mean_edge"] = float(mass[edge_idx].mean())
        res["mass_d"] = cohen(mass[core_idx], mass[edge_idx])
        res["tau32_mean_core"] = float(tau[core_idx].mean())
        res["tau32_mean_edge"] = float(tau[edge_idx].mean())
        res["tau32_d"] = cohen(tau[core_idx], tau[edge_idx])
        out["subclasses"][cname] = res
        print(cname, res, flush=True)
    with open(EXPERIMENTS / "predicted_effect_sizes.json", "w") as f:
        json.dump(out, f, indent=2)
    print("[COMPLETE]", flush=True)


if __name__ == '__main__':
    main()
