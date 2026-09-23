"""
POC: Embedding Space as a Physics Manifold
===========================================
Tests whether the ArcFace hyperspherical embedding space has a smooth,
monotonic correlation with known physics features.

If true -> the model's latent space is a genuine physics manifold
If false -> the model learned only a binary decision boundary

Experiment:
  1. Sort ALL 404K test jets by their cosine similarity to w_Top
  2. Bin into 20 equally-populated bins
  3. For each bin, compute mean mass, mean tau32, mean multiplicity
  4. Plot: do these physics features vary smoothly with cos_sim?
  
  If smooth -> embedding space encodes physics
  If step-function -> embedding space is just a classifier boundary
"""
import os, sys, time, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arcefn.utils.paths import EXPERIMENTS, DATA_DIR, CHECKPOINT
from arcefn.utils.device import get_device
from arcefn.models.efn_arcface import TopTaggingModel
from arcefn.data.loader import load_awkward, JetTaggingDataset

dev, _ = get_device()
device = dev
print(f"Device: {device}")

# Load model
model = TopTaggingModel(s=16.0, m=0.5, embedding_dim=64, particle_dim=128).to(device)
sd = torch.load(CHECKPOINT, map_location=device, weights_only=False)
model.load_state_dict(sd, strict=True)
model.eval()

# Load data
print("Loading 404K test events...")
t0 = time.time()
events, labels, weights = load_awkward(str(DATA_DIR / 'test.h5'), max_events=404000, lazy=False)
ds = JetTaggingDataset(events, labels, weights)
loader = DataLoader(ds, batch_size=1024, shuffle=False, num_workers=0)
print(f"  Data loaded in {time.time()-t0:.0f}s")

# Compute embeddings + similarities
print("Computing embeddings...")
t0 = time.time()
all_sims = []
all_masses = []
all_bools = []
with torch.no_grad():
    for x, y, w, m, _ in loader:
        x, m = x.to(device), m.to(device)
        logits, sims, emb = model(x, mask=m)
        all_sims.append(sims.cpu().numpy())
        # Truth labels
        all_bools.append(y.numpy())

sims = np.concatenate(all_sims, axis=0)  # (404K, 2)
labels = np.concatenate(all_bools, axis=0)
print(f"  Done in {time.time()-t0:.0f}s")

# Compute cos_sim_top - cos_sim_qcd (decision margin)
margin = sims[:, 1] - sims[:, 0]  # positive = top-like

# Sort by margin (the model's internal measure of "top-ness")
sort_idx = np.argsort(margin)
n_events = len(margin)

# Compute physics features via FastJet (subsample for speed)
print("Computing FastJet features (15K stratified sample)...")
t0 = time.time()

# Stratified: 7500 top-like + 7500 qcd-like
top_like = np.where(margin > 0)[0]
qcd_like = np.where(margin <= 0)[0]

n_sample = 7500
rng = np.random.RandomState(42)
idx_top = rng.choice(top_like, min(n_sample, len(top_like)), replace=False)
idx_qcd = rng.choice(qcd_like, min(n_sample, len(qcd_like)), replace=False)
sample_idx = np.concatenate([idx_top, idx_qcd])

# Compute mass for sampled events
from arcefn.utils.physics import compute_features
from arcefn.data.loader import load_events_by_indices

sample_events = load_events_by_indices(str(DATA_DIR / 'test.h5'), sample_idx.tolist())
physics_feats = compute_features(sample_events)
sample_mass = physics_feats[:, 0]
sample_margin = margin[sample_idx]
sample_labels = labels[sample_idx]
print(f"  Done in {time.time()-t0:.0f}s")

# Also get Tau32 (index 6 in physics features) and multiplicity (index 5)
sample_tau32 = physics_feats[:, 6]
sample_mult = physics_feats[:, 5]

# --- Plot 1: Physics features vs margin (binned) ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
feat_names = ['Jet Mass (GeV)', r'$\tau_{32}$', 'Multiplicity']
feat_data = [sample_mass, sample_tau32, sample_mult]

n_bins = 20
bin_edges = np.linspace(sample_margin.min(), sample_margin.max(), n_bins + 1)

for ax, feat, fname in zip(axes, feat_data, feat_names):
    bin_means = []
    bin_stds = []
    bin_centers = []
    for i in range(n_bins):
        mask_bin = (sample_margin >= bin_edges[i]) & (sample_margin < bin_edges[i+1])
        if mask_bin.sum() > 10:
            bin_means.append(feat[mask_bin].mean())
            bin_stds.append(feat[mask_bin].std() / np.sqrt(mask_bin.sum()))
            bin_centers.append((bin_edges[i] + bin_edges[i+1]) / 2)
    
    bin_centers = np.array(bin_centers)
    bin_means = np.array(bin_means)
    bin_stds = np.array(bin_stds)
    
    ax.errorbar(bin_centers, bin_means, yerr=bin_stds, fmt='o-', capsize=3, color='#2c3e50')
    ax.axvline(0, color='red', linestyle='--', alpha=0.5, label='Decision boundary')
    ax.set_xlabel('Decision margin (cos_Top - cos_QCD)', fontsize=13)
    ax.set_ylabel(fname, fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

fig.suptitle('Physics Features vs ArcFace Decision Margin (15K sampled jets)', fontsize=15, y=1.02)
fig.tight_layout()
fig.savefig(str(EXPERIMENTS / 'poc_physics_manifold.png'), dpi=150, bbox_inches='tight')
print(f"Saved: {EXPERIMENTS / 'poc_physics_manifold.png'}")

# --- Plot 2: Monotonicity test ---
# Check: is the relationship monotonic?
from scipy.stats import spearmanr

print("\n=== Monotonicity Test (Spearman correlation: margin vs physics) ===")
for feat, fname in zip(feat_data, feat_names):
    rho, p = spearmanr(sample_margin, feat)
    print(f"  {fname:<20}: rho={rho:.4f}, p={p:.2e}")

# --- Plot 3: Mass-marginalized --- 
# Show that at fixed mass, margin still has predictive power
print("\n=== Mass-Marginalized: Margin's predictive power within mass bins ===")
from sklearn.metrics import roc_auc_score

for mass_lo, mass_hi in [(0, 100), (100, 140), (140, 170), (170, 250)]:
    mask_mass = (sample_mass >= mass_lo) & (sample_mass < mass_hi)
    if mask_mass.sum() < 50:
        continue
    acc = ((sample_margin[mask_mass] > 0) == sample_labels[mask_mass]).mean()
    auc = roc_auc_score(sample_labels[mask_mass], sample_margin[mask_mass])
    n = mask_mass.sum()
    n_top = sample_labels[mask_mass].sum()
    print(f"  Mass [{mass_lo:3d},{mass_hi:3d}) GeV: N={n:5d} (Top={n_top:5d}), Acc={acc:.4f}, AUC={auc:.4f}")

# Summary
print("\n=== POC Summary ===")
print(f"  Margin-Tau32 Spearman rho: {spearmanr(sample_margin, sample_tau32)[0]:.3f}")
print(f"  Margin-Mass Spearman rho: {spearmanr(sample_margin, sample_mass)[0]:.3f}")
print(f"  Mass-only benchmark vs full model: compare AUC within bins")
print("\n[POC COMPLETE]")
