"""
Compare subclass physics: ArcFace vs Linear head
=================================================
Hypothesis: ArcFace (hypersphere S^63) produces subclasses
that match DGLAP theory; Linear (unconstrained) does not.

Output
------
* experiments/head_geometry_params.json — fit results
* experiments/head_geometry.png         — 2x4 comparison figure
"""

import json, gc, time, warnings
warnings.filterwarnings('ignore')
from collections import Counter
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path

from arcefn.utils.paths import EXPERIMENTS, DATA_DIR
from arcefn.utils.model_loading import load_model_from_dir
from arcefn.data.loader import load_awkward, JetTaggingDataset
from arcefn.utils.device import get_device
from arcefn.utils.clustering import spectral_clustering_subclass
from arcefn.utils.physics import compute_features
from arcefn.utils.figure_style import apply_style, panel_label, add_grid, save_fig, SUBCLASS_COLORS, LEGEND_KWARGS

TEST_H5 = str(DATA_DIR / 'test.h5')
N_EVENTS = 50000
BATCH_SIZE = 1024

NAME_MAP = {0: 'QCD_Core', 1: 'QCD_Edge', 2: 'Top_Core', 3: 'Top_Edge'}

def arch_name(model_dir):
    import torch
    ckpt = torch.load(str(Path(model_dir) / 'checkpoint.pt'), map_location='cpu', weights_only=False)
    if 'arcface_head.class_centers' in ckpt:
        return 'ArcFace'
    if 'classifer.weight' in ckpt or 'classifier.weight' in ckpt:
        return 'Linear'
    if 'class_centers' in ckpt:
        return 'CosLinear'
    return 'Model'

def dglap_power_law(z, A, alpha, beta):
    z = np.clip(z, 1e-10, 1 - 1e-10)
    return A * z**alpha * (1 - z)**beta

def get_embeddings(model, loader, device):
    model.eval()
    all_emb, all_labels, all_logits = [], [], []
    with torch.no_grad():
        for x, y, w, m, _ in loader:
            x, m = x.to(device), m.to(device)
            out = model(x, mask=m)
            emb = out[-1]
            all_emb.append(emb.cpu().numpy())
            all_labels.append(y.numpy())
            all_logits.append(out[0].cpu().numpy())
    return np.vstack(all_emb), np.concatenate(all_labels), np.vstack(all_logits)

def assign_subclasses(emb, labels):
    # labels here are PREDICTED classes (what the model sees), not ground truth
    sub = np.full(len(labels), -1, dtype=int)
    for cls, offset in [(0, 0), (1, 2)]:
        mask = labels == cls
        if mask.sum() < 10:
            continue
        sl, k = spectral_clustering_subclass(emb[mask], max_k=5)
        if k >= 2:
            counts = Counter(sl)
            order = sorted(counts.keys(), key=lambda c: -counts[c])
            label_map = {order[0]: 0, order[1]: 1}
            sl = np.array([label_map.get(l, 0) for l in sl])
        sl = sl % 2
        sub[mask] = sl + offset
    return sub

def compute_zg_safe(events):
    n = len(events)
    zg = np.full(n, -1.0)
    chunk = 5000
    for i in range(0, n, chunk):
        end = min(i + chunk, n)
        try:
            feats = compute_features(events[i:end])
            zg[i:end] = feats[:, 8]
        except Exception:
            pass
    return zg

def fit_subclass(zg):
    valid = (zg > 0.1) & (zg < 0.5)
    v = zg[valid]
    if len(v) < 50:
        return None
    hist, edges = np.histogram(v, bins=30, density=True)
    bc = (edges[:-1] + edges[1:]) / 2
    try:
        popt, _ = curve_fit(dglap_power_law, bc, hist, p0=[1.0, -1.0, 0.0],
                           bounds=[[0, -5, -5], [100, 10, 10]], maxfev=10000)
        res = hist - dglap_power_law(bc, *popt)
        r2 = 1 - np.sum(res**2) / (np.sum((hist - np.mean(hist))**2) + 1e-12)
        return {'A': float(popt[0]), 'alpha': float(popt[1]), 'beta': float(popt[2]),
                'R2': float(r2), 'N': int(len(v))}
    except Exception:
        return None

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Head geometry comparison (ArcFace vs Linear)")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Primary model dir (config.json + checkpoint.pt). "
                             "Default: canonical 64-d ArcFace + 64-d Linear.")
    parser.add_argument("--compare_dir", type=str, nargs='+', default=None,
                        help="One or more model dirs to compare against (only used with --model_dir). "
                             "Multiple dirs are stacked as additional rows.")
    parser.add_argument("--max_events", type=int, default=N_EVENTS,
                        help="Number of test events (default %(default)s).")
    parser.add_argument("--out_tag", type=str, default="",
                        help="Suffix for output filenames, e.g. '_coslinear' "
                             "(default: plain, matches canonical ArcFace-vs-Linear).")
    args = parser.parse_args()

    device, _ = get_device()
    print(f"Device: {device}")

    print(f"Loading {args.max_events:,} events...")
    events, labels, weights = load_awkward(TEST_H5, max_events=args.max_events)
    ds = JetTaggingDataset(events, labels, weights)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print("Computing zg (once, shared)...")
    t0 = time.time()
    zg = compute_zg_safe(events)
    print(f"  Done in {time.time()-t0:.0f}s")

    if args.model_dir:
        model_dirs = [Path(args.model_dir)]
        names = [arch_name(model_dirs[0])]
        if args.compare_dir:
            for d in args.compare_dir:
                model_dirs.append(Path(d))
                names.append(arch_name(Path(d)))
        out_dir = model_dirs[0]
    else:
        model_dirs = [EXPERIMENTS / 'robustness_s16_m05', EXPERIMENTS / 'baselines_linear_seed_42']
        names = ['ArcFace', 'Linear']
        out_dir = EXPERIMENTS

    all_results = {}
    apply_style()
    n_rows = len(model_dirs)
    fig, axes = plt.subplots(n_rows, 4, figsize=(14, 3 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]  # type: ignore[assignment]
    sc_colors = [SUBCLASS_COLORS['QCD core'], SUBCLASS_COLORS['QCD edge'],
                 SUBCLASS_COLORS['Top core'], SUBCLASS_COLORS['Top edge']]

    for row, (name, ckpt_dir) in enumerate(zip(names, model_dirs)):
        print(f"\n--- {name} ---")
        t0 = time.time()
        model = load_model_from_dir(ckpt_dir, device)
        emb, y, logits = get_embeddings(model, loader, device)
        del model; gc.collect()
        print(f"  Emb: {emb.shape} in {time.time()-t0:.0f}s")
        from arcefn.utils.pred_labels import check_cm
        preds = np.argmax(logits, axis=1)
        check_cm(preds, y, name)

        sub = assign_subclasses(emb, preds)
        n_sc = len(set(sub[sub >= 0]))
        print(f"  Subclasses: {n_sc}")

        results = {}
        for sc_id in range(4):
            mask = sub == sc_id
            n = mask.sum()
            if n < 10:
                continue
            fit = fit_subclass(zg[mask])
            sc_name = NAME_MAP[sc_id]
            if fit:
                results[sc_name] = fit
                print(f"  {sc_name}: alpha={fit['alpha']:.3f}, beta={fit['beta']:.3f}, R2={fit['R2']:.4f}, N={n}")
            else:
                print(f"  {sc_name}: fit failed, N={n}")

            # Plot
            ax = axes[row, sc_id]
            v = zg[mask]
            valid = (v > 0.1) & (v < 0.5)
            v = v[valid]
            if len(v) < 10:
                ax.text(0.5, 0.5, 'N<10', ha='center', va='center')
                continue
            ax.hist(v, bins=30, density=True, alpha=0.4, color=sc_colors[sc_id],
                   label=f'{sc_name} ($N={len(v):,}$)')
            if fit:
                z_grid = np.linspace(0.1, 0.5, 100)
                ax.plot(z_grid, dglap_power_law(z_grid, fit['A'], fit['alpha'], fit['beta']),
                       '-', color=sc_colors[sc_id], linewidth=1.2,
                       label=f'$\\alpha$={fit["alpha"]:.2f}, $R^2$={fit["R2"]:.3f}')
            ax.set_xlabel('$z_g$')
            if sc_id == 0:
                ax.set_ylabel(f'{name}\nP($z_g$)', fontsize=9)
            ax.set_title(f'{sc_name}')
            ax.legend(loc="upper right", **{**LEGEND_KWARGS, "fontsize": 6})

        all_results[name] = results

    plt.tight_layout()
    tag = args.out_tag or ""
    fig_name = f'head_geometry{tag}.png'
    json_name = f'head_geometry_params{tag}.json'
    save_fig(fig, fig_name)
    print(f"\nSaved: {fig_name}")

    with open(str(out_dir / json_name), 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved: {json_name}")

    # Print comparison table
    print(f"\n{'='*70}")
    header = f"{'Subclass':<12}"
    for nm in names:
        header += f" {nm+' a':<11} {nm+' R2':<11}"
    print(header)
    print(f"{'-'*60}")
    for sc_name in NAME_MAP.values():
        row = f"{sc_name:<12}"
        for nm in names:
            res = all_results.get(nm, {}).get(sc_name, {})
            a = f"{res.get('alpha', -1):.3f}" if res else 'N/A'
            r = f"{res.get('R2', 0):.4f}" if res else 'N/A'
            row += f" {a:<11} {r:<11}"
        print(row)
    print(f"{'='*70}")

    print("\n[DONE]")

if __name__ == '__main__':
    main()
