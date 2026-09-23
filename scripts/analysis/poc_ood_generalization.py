"""
OOD Generalization: ArcFace Cosine Similarity to Class Prototypes
================================================================
Loads 5 JetNet30 classes (g,q,w,z,t), runs through ArcFace model,
and plots the cosine similarity distributions as violin plots.

The 2D scatter of (cos_QCD, cos_Top) collapses to a curve because the
model has only 2 class centers on a hypersphere, making them anti-correlated.
Violin plots show the per-class distributions clearly.
"""
import os, sys, time, json
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from arcefn.utils.figure_style import apply_style, panel_label, add_grid, save_fig, LEGEND_KWARGS

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arcefn.utils.paths import EXPERIMENTS, CHECKPOINT, DATA_DIR, get_project_root
from arcefn.utils.device import get_device
from arcefn.models.efn_arcface import TopTaggingModel

dev, backend = get_device()
device = dev
print(f"Device: {device} ({backend})")

model = TopTaggingModel(s=16.0, m=0.5, embedding_dim=64, particle_dim=128).to(device)
sd = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
model.load_state_dict(sd, strict=True)
model.eval()
print(f"Model loaded from {CHECKPOINT}")

apply_style()

N_PAD = 200
JETS_PER_CLASS = 50000
SEED = 42

JETNET_DIR = get_project_root() / 'data' / 'jetnet'
class_config = [
    ('g', 'gluon'),
    ('q', 'light quark'),
    ('w', 'W boson'),
    ('z', 'Z boson'),
    ('t', 'top quark'),
]

def reconstruct_4momenta(jets, parts):
    N = len(jets)
    jet_pt = jets[:, 0:1]
    jet_eta = jets[:, 1:2]

    pt_part = parts[:, :, 2] * jet_pt
    eta_part = jet_eta + parts[:, :, 0]
    phi_part = parts[:, :, 1]
    mask_30 = parts[:, :, 3]

    px = pt_part * np.cos(phi_part)
    py = pt_part * np.sin(phi_part)
    pz = pt_part * np.sinh(eta_part)
    E  = pt_part * np.cosh(eta_part)

    result = np.zeros((N, N_PAD, 4), dtype=np.float32)
    mask = np.zeros((N, N_PAD), dtype=np.float32)
    result[:, :30, 0] = E
    result[:, :30, 1] = px
    result[:, :30, 2] = py
    result[:, :30, 3] = pz
    mask[:, :30] = mask_30
    return result, mask

all_cos_qcd = {}
all_cos_top = {}
all_margin = {}
total_time = 0

for jtype, cname in class_config:
    t0 = time.time()
    h5_path = JETNET_DIR / f'{jtype}.hdf5'
    if not h5_path.exists():
        print(f"  SKIP {jtype}/{cname}: {h5_path} not found")
        continue

    with h5py.File(str(h5_path), 'r') as f:
        jets = f['jet_features'][:]
        parts = f['particle_features'][:]

    n_total = len(jets)
    indices = np.random.RandomState(SEED).choice(n_total, min(JETS_PER_CLASS, n_total), replace=False)
    jets_sel = jets[indices]
    parts_sel = parts[indices]

    x_np, mask_np = reconstruct_4momenta(jets_sel, parts_sel)

    n_parts = parts_sel[:, :, 3].sum(axis=1)
    print(f"  {jtype} ({cname}): {len(indices)} jets, avg particles={n_parts.mean():.0f}")

    x_t = torch.from_numpy(x_np).to(device)
    mask_t = torch.from_numpy(mask_np).to(device)

    with torch.no_grad():
        logits, similarities, emb = model(x_t, mask=mask_t)

    cs = similarities.cpu().numpy()
    all_cos_qcd[cname] = cs[:, 0]
    all_cos_top[cname] = cs[:, 1]
    all_margin[cname] = cs[:, 1] - cs[:, 0]

    elapsed = time.time() - t0
    total_time += elapsed
    print(f"    Done in {elapsed:.0f}s")

print(f"\nTotal: {sum(len(v) for v in all_cos_qcd.values())} jets in {total_time:.0f}s")

# ── Figure: 2-panel violin plot ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
class_labels = ['gluon', 'light quark', 'W boson', 'Z boson', 'top quark']
colors = ['#377eb8', '#ff7f0e', '#2ca02c', '#e41a1c', '#984ea3']

# Panel (a): cos_sim_QCD
data_qcd = [all_cos_qcd[c] for c in class_labels]
parts_qcd = ax1.violinplot(data_qcd, positions=range(len(class_labels)),
                            showmeans=True, showmedians=False, showextrema=False)
for i, pc in enumerate(parts_qcd['bodies']):
    pc.set_facecolor(colors[i])
    pc.set_alpha(0.6)
parts_qcd['cmeans'].set_color('black')
ax1.set_xticks(range(len(class_labels)))
ax1.set_xticklabels(class_labels, rotation=15, ha='right')
ax1.set_ylabel('Cosine similarity to $W_{\\mathrm{QCD}}$')
panel_label(ax1, '(a)')
ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.4, lw=0.8)
add_grid(ax1)
ax1.set_xlim(-0.6, len(class_labels) - 0.4)

# Panel (b): cos_sim_Top
data_top = [all_cos_top[c] for c in class_labels]
parts_top = ax2.violinplot(data_top, positions=range(len(class_labels)),
                            showmeans=True, showmedians=False, showextrema=False)
for i, pc in enumerate(parts_top['bodies']):
    pc.set_facecolor(colors[i])
    pc.set_alpha(0.6)
parts_top['cmeans'].set_color('black')
ax2.set_xticks(range(len(class_labels)))
ax2.set_xticklabels(class_labels, rotation=15, ha='right')
ax2.set_ylabel('Cosine similarity to $W_{\\mathrm{Top}}$')
panel_label(ax2, '(b)')
ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.4, lw=0.8)
add_grid(ax2)
ax2.set_xlim(-0.6, len(class_labels) - 0.4)

fig.tight_layout()
save_fig(fig, 'poc_ood_similarity.png')

# Summary table
print(f"\n{'='*80}")
print(f"{'Class':<15} {'N':<6} {'cos_QCD_mean':<14} {'cos_QCD_std':<14} {'cos_Top_mean':<14} {'cos_Top_std':<14} {'margin':<10}")
print('-'*80)
for cname in class_config:
    cn = cname[1]
    if cn not in all_cos_qcd: continue
    cq = all_cos_qcd[cn]
    ct = all_cos_top[cn]
    mg = all_margin[cn]
    print(f"{cn:<15} {len(cq):<6} {cq.mean():<14.4f} {cq.std():<14.4f} "
          f"{ct.mean():<14.4f} {ct.std():<14.4f} {mg.mean():<10.4f}")

# Save results
results = {}
for cn in all_cos_qcd:
    cq = all_cos_qcd[cn]
    ct = all_cos_top[cn]
    mg = all_margin[cn]
    results[cn] = {
        'n': int(len(cq)),
        'cos_qcd_mean': float(cq.mean()),
        'cos_qcd_std': float(cq.std()),
        'cos_top_mean': float(ct.mean()),
        'cos_top_std': float(ct.std()),
        'margin_mean': float(mg.mean()),
        'margin_std': float(mg.std()),
        'frac_top': float((ct > cq).mean()),
    }

with open(str(EXPERIMENTS / 'poc_ood_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved: {EXPERIMENTS / 'poc_ood_results.json'}")
print("\n[POC COMPLETE]")
