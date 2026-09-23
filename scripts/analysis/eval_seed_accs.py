"""Evaluate test accuracy for a list of model dirs (fills sourcing gaps).

Writes experiments/seed_accs.json : {dirname: {accuracy, n}}.
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

MODEL_DIRS = [
    "baselines_coslinear_seed_42",
    "baselines_coslinear_seed_123",
    "baselines_coslinear_seed_7",
    "robustness_s22_m05",
    "robustness_s22_seed_123",
    "robustness_s22_seed_7",
]
BATCH_SIZE = 2048


def main():
    device, _ = get_device()
    events, labels, weights = load_awkward(str(DATA_DIR / 'test.h5'), lazy=False)
    out = {}
    for name in MODEL_DIRS:
        model = load_model_from_dir(EXPERIMENTS / name, device)
        model.eval()
        loader = DataLoader(JetTaggingDataset(events, labels, weights),
                            batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        correct, total = 0, 0
        with torch.no_grad():
            for x, y, w, m, _ in loader:
                x, m = x.to(device), m.to(device)
                logits = model(x, mask=m)[0]
                pred = torch.argmax(logits, dim=1).cpu().numpy()
                correct += int((pred == y.numpy()).sum())
                total += len(y)
        acc = correct / total
        out[name] = {"accuracy": acc, "n": total}
        print(f"{name}: {acc:.6f} (n={total})", flush=True)
    with open(EXPERIMENTS / "seed_accs.json", "w") as f:
        json.dump(out, f, indent=2)
    print("[COMPLETE]", flush=True)


if __name__ == '__main__':
    main()
