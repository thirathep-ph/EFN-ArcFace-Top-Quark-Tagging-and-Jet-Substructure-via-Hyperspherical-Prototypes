"""
Robustness Checkpoint Evaluation
================================
Evaluates all 9 robustness checkpoints
(scale × margin ∈ {8, 16, 32} × {0.0, 0.3, 0.5}) on the test set and
produces a results JSON and Markdown table.

Checkpoints are discovered automatically from the ``experiments/`` directory
by matching ``robustness_s{s}_m{m}/checkpoint.pt``.

Usage
-----
::

    python scripts/evaluate_robustness.py

Output
------
* ``EXPERIMENTS / robustness_results.json``
* ``EXPERIMENTS / robustness_results_table.md``
"""
import json
import time
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from arcefn.utils.paths import DATA_DIR, EXPERIMENTS
from arcefn.utils.device import get_device
from arcefn.models.efn_arcface import TopTaggingModel
from arcefn.data.loader import load_awkward, JetTaggingDataset

TEST_H5 = DATA_DIR / 'test.h5'
MAX_EVENTS = 50000


def evaluate_checkpoint(model_path, device, max_events=MAX_EVENTS):
    """
    Evaluate a single checkpoint on a subset of the test set.

    Parameters
    ----------
    model_path : str or Path
        Path to ``checkpoint.pt``.
    device : torch.device
        Compute device.
    max_events : int
        Maximum number of test events to use.

    Returns
    -------
    dict
        Keys: ``accuracy``, ``precision``, ``recall``, ``f1``,
        ``roc_auc``, ``tp``, ``fp``, ``fn``, ``tn``, ``n_events``.
    """
    model = TopTaggingModel(s=16.0, m=0.5, embedding_dim=64, particle_dim=128).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()

    events, labels, _ = load_awkward(TEST_H5, max_events=max_events)
    dataset = JetTaggingDataset(events, labels, np.ones(len(labels), dtype=np.float32))
    loader = DataLoader(dataset, batch_size=2048, shuffle=False, num_workers=0)

    all_preds = []
    all_labels = []
    all_logits = []

    with torch.no_grad():
        for x, y, w, m, _ in loader:
            x, y, m = x.to(device), y.to(device), m.to(device)
            logits, sims, _ = model(x, mask=m)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_logits.append(logits.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_logits = np.vstack(all_logits)

    acc = float(np.mean(all_preds == all_labels))
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0)

    try:
        auc = float(roc_auc_score(all_labels, all_logits[:, 1]))
    except Exception:
        auc = 0.0

    tp = int(np.sum((all_preds == 1) & (all_labels == 1)))
    fp = int(np.sum((all_preds == 1) & (all_labels == 0)))
    fn = int(np.sum((all_preds == 0) & (all_labels == 1)))
    tn = int(np.sum((all_preds == 0) & (all_labels == 0)))

    return {
        'accuracy': acc,
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'roc_auc': auc,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'n_events': len(all_labels),
    }


def main():
    """
    Discover all robustness checkpoints, evaluate each, and save results.
    """
    device, backend_type = get_device()
    print(f"Device: {device}")
    print(f"Test set: {TEST_H5}\n")

    results = []
    for entry in sorted(os.listdir(str(EXPERIMENTS))):
        if not entry.startswith("robustness_s"):
            continue
        ckpt = EXPERIMENTS / entry / "checkpoint.pt"
        if not ckpt.exists():
            print(f"  [SKIP] {entry} - no checkpoint.pt")
            continue

        parts = entry.replace("robustness_s", "").split("_m")
        try:
            s = int(parts[0])
            m = float(parts[1])
        except (ValueError, IndexError):
            print(f"  [SKIP] {entry} - cannot parse s/m")
            continue

        print(f"  [EVAL] s={s}, m={m} ({entry}) ...")
        t0 = time.time()
        res = evaluate_checkpoint(ckpt, device)
        elapsed = time.time() - t0

        res['scale'] = s
        res['margin'] = m
        res['time_seconds'] = elapsed
        results.append(res)

        print(f"    Acc={res['accuracy']:.4f}  F1={res['f1']:.4f}  "
              f"AUC={res['roc_auc']:.4f}  ({elapsed:.0f}s)")

    if not results:
        print("\nNo checkpoints found! Check experiments/ directory.")
        return

    results.sort(key=lambda x: (x['scale'], x['margin']))

    # Save JSON
    out_json = EXPERIMENTS / "robustness_results.json"
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_json}")

    # Generate Markdown
    out_md = EXPERIMENTS / "robustness_results_table.md"
    with open(out_md, 'w') as f:
        f.write("# Robustness Analysis Results\n\n")
        f.write("| Scale (s) | Margin (m) | Accuracy | Precision | Recall | "
                "F1 | ROC AUC | TP | FP | FN | TN |\n")
        f.write("|:---------:|:----------:|:--------:|:---------:|:------:|"
                ":--:|:-------:|:--:|:--:|:--:|:--:|\n")
        for r in results:
            f.write(f"| {r['scale']} | {r['margin']:.1f} | {r['accuracy']:.4f} | "
                    f"{r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | "
                    f"{r['roc_auc']:.4f} | {r['tp']:,} | {r['fp']:,} | "
                    f"{r['fn']:,} | {r['tn']:,} |\n")

        accs = [r['accuracy'] for r in results]
        f1s = [r['f1'] for r in results]
        aucs = [r['roc_auc'] for r in results]

        f.write("\n## Summary\n\n")
        f.write("| Metric | Mean | Std | Min | Max |\n")
        f.write("|:-------|:----:|:---:|:---:|:---:|\n")
        f.write(f"| Accuracy | {np.mean(accs):.4f} | {np.std(accs):.4f} | "
                f"{np.min(accs):.4f} | {np.max(accs):.4f} |\n")
        f.write(f"| F1 | {np.mean(f1s):.4f} | {np.std(f1s):.4f} | "
                f"{np.min(f1s):.4f} | {np.max(f1s):.4f} |\n")
        f.write(f"| ROC AUC | {np.mean(aucs):.4f} | {np.std(aucs):.4f} | "
                f"{np.min(aucs):.4f} | {np.max(aucs):.4f} |\n")

    print(f"Saved to {out_md}")
    print("\nDone!")


if __name__ == '__main__':
    main()
