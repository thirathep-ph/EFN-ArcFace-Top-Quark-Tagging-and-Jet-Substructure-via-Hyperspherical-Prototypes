"""
Robustness Tests: 4-Momentum Rescaling, Azimuthal Rotation, Constituent Dropout
================================================================================
Computes three robustness tests (see docs/RESULTS.md):

1. **4-Momentum Rescaling** — rescale all constituent four-momenta
   (E, px, py, pz) by factors ``[0.5, 0.8, 1.0, 1.5, 2.0]`` (a jet-energy-scale
   variation) — expected accuracy ~91.8 % invariant, by $z$-normalization
   invariance (energy fractions are invariant under a global momentum scaling).
2. **Azimuthal Rotation** — rotate constituent phi by angles
   ``[0.0, 0.1, 0.5, 1.0, 2.0]`` rad — expected accuracy ~91.8 % invariant
   (physical azimuthal rotations leave the jet-axis-relative features unchanged).
3. **Constituent Dropout** — randomly drop ``[0 %, 10 %, 20 %, 50 %, 80 %]``
   of constituents — expected graceful degradation.

Output
------
* ``EXPERIMENTS / robustness_tests.png``        — 2-panel figure
* ``EXPERIMENTS / robustness_tests_results.json`` — all numeric results
"""
import json
import time
import gc
import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path

from arcefn.utils.paths import DATA_DIR, EXPERIMENTS
from arcefn.utils.device import get_device, empty_cache
from arcefn.utils.model_loading import load_model_from_dir
from arcefn.data.loader import load_awkward, JetTaggingDataset

N_EVENTS = 404000
BATCH_SIZE = 2048

PT_SCALES = [0.5, 0.8, 1.0, 1.5, 2.0]
ROT_ANGLES = [0.0, 0.1, 0.5, 1.0, 2.0]
DROPOUT_RATES = [0.0, 0.1, 0.2, 0.5, 0.8]


def load_model(device, model_dir):
    """
    Load a trained model (any embedding dimension) from a checkpoint directory.

    Parameters
    ----------
    device : torch.device
        Device to place the model on.
    model_dir : str or Path
        Directory containing ``config.json`` and ``checkpoint.pt``.

    Returns
    -------
    torch.nn.Module
        Model in evaluation mode.
    """
    return load_model_from_dir(model_dir, device)


def load_test_data(max_events=N_EVENTS):
    """
    Load a subset of the test set as raw (non-lazy) awkward arrays.

    Returns
    -------
    events : ak.Array
        Jagged array of Momentum4D vectors.
    labels : np.ndarray
        Ground-truth labels (0 = QCD, 1 = Top).
    weights : np.ndarray
        Event weights (all ones).
    """
    test_h5 = str(DATA_DIR / 'test.h5')
    print(f"Loading {max_events:,} test events from {test_h5} ...")
    events, labels, weights = load_awkward(
        test_h5, max_events=max_events, lazy=False
    )
    print(f"  Loaded {len(labels):,} events")
    return events, labels, weights


def evaluate_accuracy(model, events, labels, device, batch_size=BATCH_SIZE):
    """
    Run the model on a set of awkward events and return classification accuracy.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model.
    events : ak.Array
        Jagged Momentum4D events.
    labels : np.ndarray
        Ground-truth labels.
    device : torch.device
        Compute device.
    batch_size : int
        Batch size for DataLoader.

    Returns
    -------
    acc : float
        Fraction of correct predictions.
    """
    ds = JetTaggingDataset(events, labels, np.ones(len(labels), dtype=np.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    correct = 0
    total = 0
    with torch.no_grad():
        for x, y, w, m, _ in loader:
            x, y, m = x.to(device), y.to(device), m.to(device)
            logits, _, _ = model(x, mask=m)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += len(y)

    return correct / total


# ── Perturbation helpers ─────────────────────────────────────────────

def rescale_4momentum(events, scale):
    """
    Rescale all constituent 4-momenta (E, px, py, pz) by *scale*.

    Parameters
    ----------
    events : ak.Array
        Jagged Momentum4D events.
    scale : float
        Multiplicative factor.

    Returns
    -------
    ak.Array
        Rescaled events.
    """
    import awkward as ak
    return ak.zip({
        'E':  events.E * scale,
        'px': events.px * scale,
        'py': events.py * scale,
        'pz': events.pz * scale,
    }, with_name="Momentum4D")


def rotate_azimuth(events, angle):
    """
    Apply a physical azimuthal rotation: phi_i -> phi_i + angle for every
    constituent (eta, pT, E unchanged). Because the EFN inputs are the
    jet-axis-relative coordinates (Delta eta, Delta phi), this
    transformation leaves the model features exactly invariant.

    Parameters
    ----------
    events : ak.Array
        Jagged Momentum4D events.
    angle : float
        Rotation angle in radians.

    Returns
    -------
    ak.Array
        Azimuthally rotated events.
    """
    import awkward as ak
    import numpy as np

    px = ak.to_numpy(ak.flatten(events.px, axis=None))
    py = ak.to_numpy(ak.flatten(events.py, axis=None))
    pz = ak.to_numpy(ak.flatten(events.pz, axis=None))
    E  = ak.to_numpy(ak.flatten(events.E, axis=None))

    pt = np.sqrt(px ** 2 + py ** 2)
    phi = np.arctan2(py, px) + angle

    px_rot = pt * np.cos(phi)
    py_rot = pt * np.sin(phi)

    counts = ak.num(events.px)
    return ak.zip({
        'E':  ak.unflatten(E, counts),
        'px': ak.unflatten(px_rot, counts),
        'py': ak.unflatten(py_rot, counts),
        'pz': ak.unflatten(pz, counts),
    }, with_name="Momentum4D")


def dropout_constituents(events, rate, rng=None):
    """
    Randomly drop a fraction of constituents from each jet.

    Parameters
    ----------
    events : ak.Array
        Jagged Momentum4D events.
    rate : float
        Fraction of constituents to drop (0.0 – 1.0).
    rng : numpy.random.Generator, optional
        RNG for reproducibility.

    Returns
    -------
    ak.Array
        Events with dropped constituents set to zero energy.
    """
    import awkward as ak
    if rng is None:
        rng = np.random.default_rng(42)

    px = ak.to_numpy(ak.flatten(events.px, axis=None))
    py = ak.to_numpy(ak.flatten(events.py, axis=None))
    pz = ak.to_numpy(ak.flatten(events.pz, axis=None))
    E  = ak.to_numpy(ak.flatten(events.E, axis=None))

    counts = np.array(ak.num(events.px))
    n_total = len(E)

    # Generate per-constituent keep mask
    keep = np.ones(n_total, dtype=bool)
    ci = 0
    for n in counts:
        if n > 0 and rate > 0:
            n_drop = max(1, int(round(n * rate)))
            drop_idx = rng.choice(n, n_drop, replace=False)
            keep[ci + drop_idx] = False
        ci += n

    px[~keep] = 0.0
    py[~keep] = 0.0
    pz[~keep] = 0.0
    E[~keep] = 0.0

    return ak.zip({
        'E':  ak.unflatten(E, counts),
        'px': ak.unflatten(px, counts),
        'py': ak.unflatten(py, counts),
        'pz': ak.unflatten(pz, counts),
    }, with_name="Momentum4D")


# ── Main ────────────────────────────────────────────────────────────

def main():
    """
    Run all three robustness tests and save results + figure.

    Results are saved to:
        * ``<out_dir> / robustness_tests_results.json``
        * ``<out_dir> / robustness_tests.png``
    where ``<out_dir>`` is ``--model_dir`` if given, else ``EXPERIMENTS``.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Robustness tests (rescaling, rotation, dropout)")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Model directory (config.json + checkpoint.pt). "
                             "Default: canonical MODEL_DIR.")
    parser.add_argument("--max_events", type=int, default=N_EVENTS,
                        help="Number of test events (default %(default)s).")
    args = parser.parse_args()

    model_dir = Path(args.model_dir) if args.model_dir else None
    out_dir = model_dir if model_dir else EXPERIMENTS

    print("=" * 60)
    print("Robustness Tests")
    print("=" * 60)

    device, backend_type = get_device()
    model = load_model(device, model_dir)
    events, labels, _ = load_test_data(args.max_events)

    results = {}

    # ── 1. 4-Momentum Rescaling ─────────────────────────────────
    print("\n--- 4-Momentum Rescaling ---")
    pt_accs = []
    pt_times = []
    for scale in PT_SCALES:
        t0 = time.time()
        scaled = rescale_4momentum(events, scale)
        acc = evaluate_accuracy(model, scaled, labels, device)
        elapsed = time.time() - t0
        pt_accs.append(acc)
        pt_times.append(elapsed)
        print(f"  scale={scale:.1f}: acc={acc:.4f}  ({elapsed:.1f}s)")
    results['pt_rescaling'] = {
        'scales': PT_SCALES,
        'accuracies': pt_accs,
        'times_seconds': pt_times,
    }

    # ── 2. Jet Rotation ──────────────────────────────────────────
    print("\n--- Azimuthal Rotation ---")
    rot_accs = []
    rot_times = []
    for angle in ROT_ANGLES:
        t0 = time.time()
        rotated = rotate_azimuth(events, angle)
        acc = evaluate_accuracy(model, rotated, labels, device)
        elapsed = time.time() - t0
        rot_accs.append(acc)
        rot_times.append(elapsed)
        print(f"  angle={angle:.1f} rad: acc={acc:.4f}  ({elapsed:.1f}s)")
    results['jet_rotation'] = {
        'angles_rad': ROT_ANGLES,
        'accuracies': rot_accs,
        'times_seconds': rot_times,
    }

    # ── 3. Constituent Dropout ───────────────────────────────────
    print("\n--- Constituent Dropout ---")
    rng = np.random.default_rng(42)
    drop_accs = []
    drop_times = []
    for rate in DROPOUT_RATES:
        t0 = time.time()
        dropped = dropout_constituents(events, rate, rng)
        acc = evaluate_accuracy(model, dropped, labels, device)
        elapsed = time.time() - t0
        drop_accs.append(acc)
        drop_times.append(elapsed)
        print(f"  dropout={rate:.0%}: acc={acc:.4f}  ({elapsed:.1f}s)")
    results['constituent_dropout'] = {
        'dropout_rates': DROPOUT_RATES,
        'accuracies': drop_accs,
        'times_seconds': drop_times,
    }

    # ── Save JSON ────────────────────────────────────────────────
    out_json = out_dir / 'robustness_tests_results.json'
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_json}")

    # ── 2-Panel Figure ───────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: pT Rescaling
    ax1.axhline(y=pt_accs[2], color='gray', linestyle='--', alpha=0.5,
                label=f'Baseline ({pt_accs[2]:.3f})')
    ax1.plot(PT_SCALES, pt_accs, 'bo-', markersize=6)
    ax1.set_xlabel('pT Scale Factor')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('pT Rescaling Robustness')
    ax1.set_xscale('log', base=2)
    ax1.set_xticks(PT_SCALES)
    ax1.set_xticklabels([f'{s:.1f}' for s in PT_SCALES])
    ax1.set_ylim(0.75, 1.0)
    ax1.grid(True, alpha=0.15)
    ax1.legend()

    # Right: Constituent Dropout
    ax2.axhline(y=drop_accs[0], color='gray', linestyle='--', alpha=0.5,
                label=f'Baseline ({drop_accs[0]:.3f})')
    ax2.plot(DROPOUT_RATES, drop_accs, 'rs-', markersize=6)
    ax2.set_xlabel('Dropout Rate')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Constituent Dropout Robustness')
    ax2.set_ylim(0.3, 1.0)
    ax2.grid(True, alpha=0.15)
    ax2.legend()

    plt.tight_layout()
    out_fig = out_dir / 'robustness_tests.png'
    plt.savefig(out_fig, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to {out_fig}")

    print("\n[COMPLETE]")


if __name__ == '__main__':
    main()
