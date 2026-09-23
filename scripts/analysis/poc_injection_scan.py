"""
POC: Injection Scan — Prototype Behavior vs Signal Fraction
============================================================
Tests whether the model's classification calibration and prototype geometry
shift when the Top signal fraction is systematically varied from 0% to 100%.

Key question: Is the model well-calibrated across all class balances?
Does the margin distribution change with composition?
"""
import os, sys, time, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arcefn.utils.paths import EXPERIMENTS, DATA_DIR, MODEL_DIR
from arcefn.utils.device import get_device
from arcefn.data.loader import load_awkward
from arcefn.models.efn_arcface import TopTaggingModel

dev, backend = get_device()
device = dev
print(f"Device: {device} ({backend})")

checkpoint_path = MODEL_DIR / 'checkpoint.pt'
print(f"Loading model from {checkpoint_path}")
model = TopTaggingModel(s=16.0, m=0.5, embedding_dim=64, particle_dim=128).to(device)
sd = torch.load(checkpoint_path, map_location=device, weights_only=False)
model.load_state_dict(sd, strict=True)
model.eval()

print("\nLoading full test set (404K)...")
events, labels, weights = load_awkward(DATA_DIR / 'test.h5', max_events=404000, lazy=False)
print(f"Loaded {len(labels)} events ({labels.sum():.0f} Top, {((1-labels)).sum():.0f} QCD)")

print("Computing embeddings...")
from torch.utils.data import DataLoader
from arcefn.data.loader import JetTaggingDataset
ds = JetTaggingDataset(events, labels, weights)
loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)

all_logits = []
all_labels = []
all_margins = []
with torch.no_grad():
    for x, y, w, m, _ in loader:
        x, m = x.to(device), m.to(device)
        logits, similarities, emb = model(x, mask=m)
        margin = similarities[:, 1] - similarities[:, 0]
        all_logits.append(logits.cpu().numpy())
        all_labels.append(y.cpu().numpy())
        all_margins.append(margin.cpu().numpy())

all_logits = np.concatenate(all_logits)
all_labels = np.concatenate(all_labels)
all_margins = np.concatenate(all_margins)
print(f"Embeddings shape: {all_logits.shape}")

signal_fracs = np.arange(0.0, 1.05, 0.05)
n_per_bin = 20000

results = []
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for sf in signal_fracs:
    n_sig = int(n_per_bin * sf)
    n_bkg = n_per_bin - n_sig

    sig_mask = all_labels == 1
    bkg_mask = all_labels == 0

    sig_idx = np.where(sig_mask)[0]
    bkg_idx = np.where(bkg_mask)[0]

    rng = np.random.RandomState(42)
    chosen_sig = rng.choice(sig_idx, n_sig, replace=False) if n_sig > 0 else np.array([], dtype=int)
    chosen_bkg = rng.choice(bkg_idx, n_bkg, replace=False)

    idx = np.concatenate([chosen_sig, chosen_bkg])
    np.random.shuffle(idx)

    batch_labels = all_labels[idx]
    batch_margins = all_margins[idx]
    pred = (batch_margins > 0).astype(float)
    true_top_frac = batch_labels.mean()
    pred_top_frac = pred.mean()
    acc = (pred == batch_labels).mean()

    margin_top = batch_margins[batch_labels == 1]
    margin_qcd = batch_margins[batch_labels == 0]

    results.append({
        'true_signal_frac': float(true_top_frac),
        'pred_signal_frac': float(pred_top_frac),
        'accuracy': float(acc),
        'margin_mean_top': float(margin_top.mean()) if len(margin_top) > 0 else 0,
        'margin_mean_qcd': float(margin_qcd.mean()) if len(margin_qcd) > 0 else 0,
        'margin_std_top': float(margin_top.std()) if len(margin_top) > 0 else 0,
        'margin_std_qcd': float(margin_qcd.std()) if len(margin_qcd) > 0 else 0,
    })

true_sf = [r['true_signal_frac'] for r in results]
pred_sf = [r['pred_signal_frac'] for r in results]
acc_means = [r['accuracy'] for r in results]
margins_top = [r['margin_mean_top'] for r in results]
margins_qcd = [r['margin_mean_qcd'] for r in results]

axes[0].plot(true_sf, pred_sf, 'bo-', markersize=6, label='Predicted')
axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
axes[0].set_xlabel('True Signal Fraction', fontsize=13)
axes[0].set_ylabel('Predicted Signal Fraction', fontsize=13)
axes[0].set_title('Calibration Curve', fontsize=14)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(-0.02, 1.02)
axes[0].set_ylim(-0.02, 1.02)
axes[0].set_aspect('equal')

axes[1].plot(true_sf, acc_means, 'rs-', markersize=6)
axes[1].axhline(0.9184, color='gray', linestyle=':', alpha=0.5, label='Full test set (50:50)')
axes[1].set_xlabel('True Signal Fraction', fontsize=13)
axes[1].set_ylabel('Accuracy', fontsize=13)
axes[1].set_title('Accuracy vs Signal Fraction', fontsize=14)
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=11)

axes[2].plot(true_sf, margins_top, marker='^', markersize=6, label='Top (signal)', color='#e41a1c')
axes[2].plot(true_sf, margins_qcd, marker='s', markersize=6, label='QCD (background)', color='#377eb8')
axes[2].axhline(0, color='gray', linestyle='--', alpha=0.5)
axes[2].set_xlabel('True Signal Fraction', fontsize=13)
axes[2].set_ylabel('Mean Margin', fontsize=13)
axes[2].set_title('Margin vs Signal Fraction', fontsize=14)
axes[2].legend(fontsize=11)
axes[2].grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(str(EXPERIMENTS / 'poc_injection_scan.png'), dpi=150, bbox_inches='tight')
print(f"\nSaved: {EXPERIMENTS / 'poc_injection_scan.png'}")

print(f"\n{'TrueFrac':<10} {'PredFrac':<10} {'Accuracy':<10} {'MarginTop':<12} {'MarginQCD':<12}")
print('-' * 54)
for r in results:
    print(f"{r['true_signal_frac']:<10.3f} {r['pred_signal_frac']:<10.3f} "
          f"{r['accuracy']:<10.4f} {r['margin_mean_top']:<12.4f} {r['margin_mean_qcd']:<12.4f}")

with open(str(EXPERIMENTS / 'poc_injection_scan_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

ece = np.mean([abs(r['true_signal_frac'] - r['pred_signal_frac']) for r in results])
print(f"\nExpected Calibration Error (ECE): {ece:.4f}")
print("[POC COMPLETE]")
