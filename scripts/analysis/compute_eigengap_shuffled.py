"""Eigengap k* measurement on the label-shuffled control model.

Same pipeline as compute_eigengap.py but loads experiments/retrain_shuffled
and writes outputs there (never touches canonical artifacts).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from arcefn.utils.paths import DATA_DIR, EXPERIMENTS
from arcefn.utils.device import get_device
from arcefn.utils.model_loading import load_model_from_dir
from arcefn.data.loader import load_awkward, JetTaggingDataset
from arcefn.utils.clustering import estimate_clusters_eigengap
from arcefn.utils.figure_style import apply_style, panel_label, add_grid, save_fig, COLORS, LEGEND_KWARGS

MODEL_DIR = EXPERIMENTS / "retrain_shuffled"
MAX_EVENTS = 80000
BATCH_SIZE = 2048


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default=str(MODEL_DIR))
    p.add_argument("--tag", default="shuffled")
    p.add_argument("--max_events", type=int, default=MAX_EVENTS)
    p.add_argument("--start_event", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    tag = args.tag
    device, _ = get_device()
    model = load_model_from_dir(model_dir, device)
    model.eval()

    events, labels, weights = load_awkward(DATA_DIR / 'test.h5', max_events=args.max_events, start_event=args.start_event, lazy=False)
    loader = DataLoader(JetTaggingDataset(events, labels, weights),
                        batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    all_emb, all_preds = [], []
    with torch.no_grad():
        for x, y, w, m, _ in loader:
            x, m = x.to(device), m.to(device)
            out = model(x, mask=m)
            logits, jet_emb = (out[0], out[-1])  # ArcFace returns 3-tuple, CosLinear/Linear 2-tuple
            all_emb.append(F.normalize(jet_emb, p=2, dim=1).cpu().numpy())
            all_preds.append(torch.argmax(logits, dim=1).cpu().numpy())
    embeddings = np.concatenate(all_emb)
    preds = np.concatenate(all_preds)
    n0, n1 = int((preds == 0).sum()), int((preds == 1).sum())
    print(f"pred split: QCD={n0}, Top={n1}", flush=True)

    results = {"pred_split": {"QCD": n0, "Top": n1}}
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for idx, (name, val) in enumerate([("QCD", 0), ("Top", 1)]):
        ax = axes[idx]
        X = embeddings[preds == val]
        if len(X) < 50:
            results[name] = {"optimal_k": None, "n_events": len(X), "note": "collapsed: too few events"}
            ax.text(0.5, 0.5, f"collapsed (n={len(X)})", ha="center", transform=ax.transAxes)
            continue
        k_opt, evals = estimate_clusters_eigengap(X, max_k=10)
        n_show = min(8, len(evals))
        ks = np.arange(1, n_show + 1)
        color = COLORS['qcd'] if name == "QCD" else COLORS['top']
        ax.plot(ks, evals[:n_show], 'o-', color=color, markersize=5, linewidth=1.2)
        ax.axvline(k_opt, color='0.3', linestyle='--', linewidth=0.8, label=f'Optimal $k$ = {k_opt}')
        gaps = np.diff(evals[:n_show])
        ax2 = ax.inset_axes([0.35, 0.35, 0.6, 0.55])
        ax2.bar(np.arange(2, n_show + 1), gaps, width=0.6, color='0.65', alpha=0.8)
        ax2.axvline(k_opt, color='0.3', linestyle='--', linewidth=0.8)
        ax.set_xlabel('Eigenvalue index $k$')
        ax.set_ylabel(r'$\lambda_k$')
        ax.set_xticks(ks)
        ax.legend(loc='lower right', **LEGEND_KWARGS)
        add_grid(ax)
        panel_label(ax, f"({chr(ord('a') + idx)})")
        results[name] = {"optimal_k": int(k_opt), "n_events": len(X),
                         "eigenvalues": [float(v) for v in evals]}
        print(f"{name}: n={len(X)}, k*={k_opt}", flush=True)
    plt.tight_layout()
    fig.savefig(model_dir / f'eigengap_{tag}.png', dpi=200, bbox_inches='tight')
    with open(model_dir / f"eigengap_{tag}.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results:", json.dumps({k: v.get("optimal_k") for k, v in results.items() if k != "pred_split"}), flush=True)
    print("[COMPLETE]", flush=True)


if __name__ == '__main__':
    main()
