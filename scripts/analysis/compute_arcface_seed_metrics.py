"""ArcFace seed-to-seed classification metrics (test set).

Trains nothing: loads each ArcFace seed checkpoint (42, 123, 7), runs a single
test-set forward pass, and reports test accuracy, confusion matrix, precision,
recall, and F1. Also reads ``history.json`` for the best/final validation
accuracy, so seed variability can be quoted on the same footing as the
baseline seeds in Table 1.

Artifacts
---------
``experiments/arcface_seed_metrics.json``
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from arcefn.data.loader import load_awkward, JetTaggingDataset
from arcefn.models.efn_arcface import TopTaggingModel
from arcefn.utils.device import get_device
from arcefn.utils.paths import DATA_DIR, EXPERIMENTS

MAX_EVENTS = 404000
BATCH_SIZE = 2048

SEED_DIRS = {
    42: EXPERIMENTS / "robustness_s16_m05",
    123: EXPERIMENTS / "arcface_seed_123",
    7: EXPERIMENTS / "arcface_seed_7",
}


def test_predictions(seed: int, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Return (true labels, predicted labels) for ``seed`` on the full test set."""
    ckpt = SEED_DIRS[seed] / "checkpoint.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"no checkpoint for seed {seed}: {ckpt}")

    model = TopTaggingModel(s=16.0, m=0.5, embedding_dim=64, particle_dim=128).to(device)
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    sd = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    model.load_state_dict(sd, strict=True)
    model.eval()

    events, labels, weights = load_awkward(
        str(DATA_DIR / "test.h5"), max_events=MAX_EVENTS, lazy=False
    )
    ds = JetTaggingDataset(events, labels, weights)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    preds = []
    with torch.no_grad():
        for x, y, w, m, _ in loader:
            x, m = x.to(device), m.to(device)
            logits, _, _ = model(x, mask=m)
            preds.append(logits.argmax(dim=1).cpu().numpy())
    preds = np.concatenate(preds)
    return labels, preds


def val_accs(seed: int) -> dict:
    hist = json.loads((SEED_DIRS[seed] / "history.json").read_text())
    va = np.asarray(hist["val_acc"])
    return {
        "best_val_acc": float(va.max()),
        "best_val_epoch": int(va.argmax()) + 1,
        "final_val_acc": float(va[-1]),
    }


def main() -> None:
    t0 = time.time()
    device, _ = get_device()

    out = {}
    for seed in (42, 123, 7):
        print(f"=== Seed {seed} ===")
        labels, preds = test_predictions(seed, device)
        n = len(labels)
        tp = int(np.sum((preds == 1) & (labels == 1)))
        tn = int(np.sum((preds == 0) & (labels == 0)))
        fp = int(np.sum((preds == 1) & (labels == 0)))
        fn = int(np.sum((preds == 0) & (labels == 1)))
        acc = (tp + tn) / n
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        out[seed] = {
            "accuracy": round(float(acc), 6),
            "confusion_matrix": [[tn, fp], [fn, tp]],
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "n_events": n,
            **val_accs(seed),
        }
        print(f"  test acc={acc:.6f}  cm=[[{tn},{fp}],[{fn},{tp}]]")
        print(f"  best_val={out[seed]['best_val_acc']:.6f} (ep {out[seed]['best_val_epoch']})"
              f"  final_val={out[seed]['final_val_acc']:.6f}")

    out["elapsed_seconds"] = round(time.time() - t0, 1)
    path = EXPERIMENTS / "arcface_seed_metrics.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {path}  ({out['elapsed_seconds']}s)")


if __name__ == "__main__":
    main()
