"""Verify misc paper numbers: m>160 QCD fraction + full-split EDA moments.

Saves experiments/misc_verify.json
"""
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from arcefn.utils.paths import DATA_DIR, EXPERIMENTS
from arcefn.utils.device import get_device
from arcefn.utils.model_loading import load_model_from_dir
from arcefn.data.loader import load_awkward, JetTaggingDataset
import awkward as ak

BATCH_SIZE = 2048


def split_moments(name):
    events, labels, weights = load_awkward(str(DATA_DIR / f'{name}.h5'), lazy=False)
    jets = ak.sum(events, axis=1)
    pt = np.asarray(np.sqrt(jets.px**2 + jets.py**2))
    mass = np.asarray(jets.mass)
    ncons = np.asarray(ak.num(events))
    eta = np.asarray(jets.eta)
    out = {"n": int(len(events)), "pt_mean": float(pt.mean()),
           "mass_mean": float(mass.mean()), "eta_mean": float(np.abs(eta).mean()),
           "ncons_mean": float(ncons.mean()),
           "pt_std": float(pt.std()), "mass_std": float(mass.std())}
    print(name, out, flush=True)
    return out


def main():
    out = {"eda_splits": {s: split_moments(s) for s in ["train", "val", "test"]}}
    device, _ = get_device()
    model = load_model_from_dir(EXPERIMENTS / "robustness_s16_m05", device)
    model.eval()
    events, labels, weights = load_awkward(str(DATA_DIR / 'test.h5'), lazy=False)
    loader = DataLoader(JetTaggingDataset(events, labels, weights),
                        batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    preds = []
    with torch.no_grad():
        for x, y, w, m, _ in loader:
            x, m = x.to(device), m.to(device)
            preds.append(torch.argmax(model(x, mask=m)[0], dim=1).cpu().numpy())
    preds = np.concatenate(preds)
    feat = np.load(EXPERIMENTS / "rf_cache_test.npz")["features"]
    mass = feat[:, 0]
    hi = mass > 160
    out["mass_gt160"] = {"n": int(hi.sum()),
                         "frac_pred_qcd": float((preds[hi] == 0).mean())}
    print(out["mass_gt160"], flush=True)
    with open(EXPERIMENTS / "misc_verify.json", "w") as f:
        json.dump(out, f, indent=2)
    print("[COMPLETE]", flush=True)


if __name__ == '__main__':
    main()
