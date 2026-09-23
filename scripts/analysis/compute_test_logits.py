"""Compute main-model logits for the full 404k test set (aligned with rf_cache_test.npz).

Reproduces the exact event selection of ``compute_rf_baseline.load_features``
(seed-42 sorted subsample when total > 404000, else arange) so rows align
positionally with ``experiments/rf_cache_test.npz`` features/labels, the
embeddings of ``robustness_s16_m05/embeddings.npz``, and the subclass labels
of ``physics_sr/subclass_labels.npz``.

Alignment is verified: loaded labels must equal rf_cache_test labels exactly.

Run: python -m scripts.analysis.compute_test_logits
Artifact: experiments/physics_sr/logits.npz (logit_top, P_top, labels)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from arcefn.data.loader import load_awkward
from arcefn.models.efn_arcface import TopTaggingModel
from arcefn.utils.device import get_device
from arcefn.utils.paths import CHECKPOINT, DATA_DIR, EXPERIMENTS

MAX_EVENTS = 404000
BATCH = 512
SEED = 42
OUT_DIR = EXPERIMENTS / "physics_sr"


def main() -> None:
    t0 = time.time()
    dev, _ = get_device()

    model = TopTaggingModel(s=16.0, m=0.5, embedding_dim=64, particle_dim=128)
    ck = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    model.load_state_dict(sd)
    model.to(dev)
    model.eval()

    events, labels, _ = load_awkward(str(DATA_DIR / "test.h5"), max_events=MAX_EVENTS, lazy=True)
    total = len(labels)
    n_use = min(MAX_EVENTS, total)
    if n_use < total:
        rng = np.random.default_rng(SEED)
        idx = np.sort(rng.choice(total, n_use, replace=False))
    else:
        idx = np.arange(total)
    print(f"[logits] n_use={n_use} of total={total}")

    rf = np.load(EXPERIMENTS / "rf_cache_test.npz")
    labels_rf = rf["labels"]
    assert len(labels_rf) == n_use, f"cache has {len(labels_rf)} rows, expected {n_use}"
    assert np.array_equal(labels_rf, labels[idx]), "label mismatch -> alignment broken"
    print("[logits] alignment with rf_cache_test.npz verified (labels identical)")

    E = np.asarray(events.E, dtype=np.float32)[idx]
    px = np.asarray(events.px, dtype=np.float32)[idx]
    py = np.asarray(events.py, dtype=np.float32)[idx]
    pz = np.asarray(events.pz, dtype=np.float32)[idx]
    labels = labels[idx].astype(np.int32)
    mask = (E > 0).astype(np.float32)
    X = np.stack([E, px, py, pz], axis=-1)

    jet_p4 = np.sum(X * mask[..., None], axis=1)
    jet_pt = np.sqrt(jet_p4[:, 1] ** 2 + jet_p4[:, 2] ** 2).astype(np.float32)
    jet_mass2 = jet_p4[:, 3] ** 2 - jet_p4[:, 1] ** 2 - jet_p4[:, 2] ** 2 - jet_p4[:, 0] ** 2
    jet_mass = np.sqrt(np.maximum(jet_mass2, 0.0)).astype(np.float32)

    logits = np.zeros((n_use, 2), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, n_use, BATCH):
            e = min(s + BATCH, n_use)
            out = model(torch.from_numpy(X[s:e]).to(dev),
                        mask=torch.from_numpy(mask[s:e]).to(dev))
            logits[s:e] = out[0].cpu().numpy()
    P = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    logitdiff = logits[:, 1] - logits[:, 0]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_DIR / "logits.npz",
        logit_top=logits[:, 1].astype(np.float32),
        logit_qcd=logits[:, 0].astype(np.float32),
        logitdiff=logitdiff.astype(np.float32),
        P_top=P[:, 1].astype(np.float32),
        labels=labels,
        jet_pt=jet_pt,
        jet_mass=jet_mass,
    )
    acc = float(np.mean((P[:, 1] > 0.5).astype(int) == labels))
    acc_d = float(np.mean((logitdiff > 0).astype(int) == labels))
    assert abs(acc - acc_d) < 1e-5, "logitdiff inconsistent with P_top"
    print(f"[logits] saved {OUT_DIR / 'logits.npz'} in {time.time() - t0:.1f}s | "
          f"P_top>0.5 acc = {acc:.5f} (= logitdiff>0 acc {acc_d:.5f})")


if __name__ == "__main__":
    main()
