"""Adversarial robustness tests (FGSM and PGD) on the ArcFace EFN checkpoint.

Reproduces the Table 3 adversarial rows with a reproducible, seeded setup:
- 20k test events (seeded random subsample, seed 42)
- FGSM (eps=0.01, one step) and PGD (eps=0.01, alpha=0.002, 10 steps, L-inf ball)
- Perturbations are masked to real constituents; energy is clamped to be
  non-negative (physical constraint); padding remains zero.

Usage:
    python -m scripts.analysis.compute_adversarial_tests [--max-events 20000]
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from arcefn.data.loader import get_h5_len, load_awkward
from arcefn.models.efn_arcface import TopTaggingModel
from arcefn.utils.device import get_device
from arcefn.utils.paths import DATA_DIR, EXPERIMENTS

TEST_H5 = str(DATA_DIR / "test.h5")
CHECKPOINT = EXPERIMENTS / "robustness_s16_m05" / "checkpoint.pt"
RESULTS_PATH = EXPERIMENTS / "adversarial_results.json"


def load_data(max_events: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    total = get_h5_len(TEST_H5)
    rng = np.random.default_rng(seed)
    index = np.sort(rng.choice(total, max_events, replace=False))
    chunk = 100_000
    parts: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    for c0 in range(0, total, chunk):
        events, labels, _ = load_awkward(
            TEST_H5, start_event=c0, max_events=min(chunk, total - c0), lazy=True
        )
        sel = index[(index >= c0) & (index < c0 + len(labels))] - c0
        if len(sel) == 0:
            continue
        E = np.asarray(events.E, dtype=np.float32)[sel]
        px = np.asarray(events.px, dtype=np.float32)[sel]
        py = np.asarray(events.py, dtype=np.float32)[sel]
        pz = np.asarray(events.pz, dtype=np.float32)[sel]
        parts.append(np.stack([E, px, py, pz], axis=-1))
        labels_all.append(labels[sel])
    x = np.concatenate(parts, axis=0)
    return x, np.concatenate(labels_all).astype(np.int64)


def evaluate(model: torch.nn.Module, x: torch.Tensor, mask: torch.Tensor,
             labels: torch.Tensor, batch_size: int = 1024) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            xb, mb, yb = x[i:i + batch_size], mask[i:i + batch_size], labels[i:i + batch_size]
            logits, _, _ = model(xb, mask=mb)
            correct += (logits.argmax(dim=1) == yb).sum().item()
    return correct / x.shape[0]


def attack(model: torch.nn.Module, x0: torch.Tensor, mask: torch.Tensor,
           labels: torch.Tensor, eps: float, alpha: float, steps: int,
           batch_size: int = 1024) -> torch.Tensor:
    """PGD with L-inf budget eps; FGSM corresponds to steps=1 with alpha=eps."""
    model.eval()
    x_adv = x0.clone()
    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        losses = []
        for i in range(0, x0.shape[0], batch_size):
            xb, mb, yb = x_adv[i:i + batch_size], mask[i:i + batch_size], labels[i:i + batch_size]
            logits, _, _ = model(xb, mask=mb)
            losses.append(F.cross_entropy(logits, yb))
        loss = torch.stack(losses).sum()
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv + alpha * grad.sign()
        x_adv = torch.clamp(x_adv, x0 - eps, x0 + eps)
        x_adv = x_adv * mask[..., None].float()
        x_adv[..., 0] = torch.clamp(x_adv[..., 0], min=0.0)
    return x_adv.detach()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-events", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eps", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.002)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    device, _ = get_device()
    model = TopTaggingModel(s=16.0, m=0.5, embedding_dim=64, particle_dim=128)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device, weights_only=False))
    model.to(device)

    x, labels = load_data(args.max_events, args.seed)
    x_t = torch.from_numpy(x).to(device)
    mask_t = (x_t[:, :, 0] > 0).float()
    y_t = torch.from_numpy(labels).to(device)

    clean_acc = evaluate(model, x_t, mask_t, y_t)
    x_fgsm = attack(model, x_t, mask_t, y_t, args.eps, args.eps, 1)
    fgsm_acc = evaluate(model, x_fgsm, mask_t, y_t)
    x_pgd = attack(model, x_t, mask_t, y_t, args.eps, args.alpha, args.steps)
    pgd_acc = evaluate(model, x_pgd, mask_t, y_t)

    results = {
        "n_events": args.max_events,
        "seed": args.seed,
        "eps": args.eps,
        "pgd_alpha": args.alpha,
        "pgd_steps": args.steps,
        "clean_accuracy": clean_acc,
        "fgsm_accuracy": fgsm_acc,
        "fgsm_drop": clean_acc - fgsm_acc,
        "pgd_accuracy": pgd_acc,
        "pgd_drop": clean_acc - pgd_acc,
        "checkpoint": str(CHECKPOINT),
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"Saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
