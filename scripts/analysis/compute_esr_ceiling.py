"""Check the oracle ceiling of the 2D-plane distillation path.

If the LS-plane (jet128 @ L -> (cos_QCD, cos_Top)) alone cannot reach the
model's accuracy, then even a PERFECT symbolic phi_sym (zero approximation
error) cannot either — the plane path is lossy by construction (R2=0.677)
and the readout G must absorb the rest.

Computes, on the 40k seed-42 esr subset:
  - NN model softmax accuracy (ground truth for distillation)
  - LS-plane accuracy with score = s * (coshat_Top - coshat_QCD)
  - LS-plane AUC
  - correlation of plane score with NN logit difference
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from arcefn.data.loader import load_awkward
from arcefn.models.efn_arcface import TopTaggingModel
from arcefn.utils.device import get_device
from arcefn.utils.paths import CHECKPOINT, DATA_DIR, EXPERIMENTS

S = 16.0


def main() -> None:
    dev, _ = get_device()
    model = TopTaggingModel(s=16.0, m=0.5, embedding_dim=64, particle_dim=128)
    ck = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    model.load_state_dict(sd)
    model.to(dev)
    model.eval()

    events, labels, _ = load_awkward(str(DATA_DIR / "test.h5"), max_events=40_000, lazy=True)
    E = np.asarray(events.E, dtype=np.float32)
    px = np.asarray(events.px, dtype=np.float32)
    py = np.asarray(events.py, dtype=np.float32)
    pz = np.asarray(events.pz, dtype=np.float32)
    labels = labels.astype(np.int32)
    mask = (E > 0).astype(np.float32)

    X = np.stack([E, px, py, pz], axis=-1)
    pt = np.sqrt(px**2 + py**2 + 1e-10)
    eta = np.arcsinh(pz / (pt + 1e-10))
    phi = np.arctan2(py, px)
    jet_p4 = (X * mask[..., None]).sum(axis=1)
    j_pt = np.sqrt(jet_p4[:, 1] ** 2 + jet_p4[:, 2] ** 2 + 1e-10)
    j_eta = np.arcsinh(jet_p4[:, 3] / (j_pt + 1e-10))
    j_phi = np.arctan2(jet_p4[:, 2], jet_p4[:, 1])
    d_eta = eta - j_eta[:, None]
    d_phi = phi - j_phi[:, None]
    d_phi = (d_phi + np.pi) % (2 * np.pi) - np.pi
    coords = np.stack([d_eta, d_phi], axis=-1)
    z = E / (E.sum(axis=1, keepdims=True) + 1e-10)
    z = z * mask

    coords_t = torch.from_numpy(coords).to(dev)
    logits_all = []
    phi_all = []
    with torch.no_grad():
        for s0 in range(0, len(labels), 512):
            e = min(s0 + 512, len(labels))
            c = coords_t[s0:e]
            fi = model.efn.phi(c.reshape(-1, 2)).reshape(e - s0, -1, 128)
            phi_all.append(fi.cpu().numpy())
            out = model(torch.from_numpy(X[s0:e]).to(dev),
                        mask=torch.from_numpy(mask[s0:e]).to(dev))
            logits_all.append(out[0].cpu().numpy())
    phi_net = np.concatenate(phi_all, axis=0)
    logits = np.concatenate(logits_all, axis=0)

    # LS plane projector (same as compute_esr_targets)
    cos = (logits / S).astype(np.float64)
    jet128 = (phi_net * z[:, :, None]).sum(axis=1)
    L, *_ = np.linalg.lstsq(jet128.astype(np.float64), cos, rcond=None)
    cos_hat = jet128 @ L
    r2 = 1.0 - float(np.sum((cos - cos_hat) ** 2) / np.sum((cos - cos.mean(0)) ** 2))

    # NN accuracy
    nn_pred = logits.argmax(1)
    nn_acc = float((nn_pred == labels).mean())

    # plane accuracy: score = s*(coshat_top - coshat_qcd)
    score = S * (cos_hat[:, 1] - cos_hat[:, 0])
    plane_acc = float(((score > 0).astype(int) == labels).mean())
    plane_auc = float(roc_auc_score(labels, score))

    # plane score vs NN logit-diff correlation (ceiling of G)
    nn_diff = logits[:, 1] - logits[:, 0]
    corr = float(np.corrcoef(score, nn_diff)[0, 1])

    # accuracy using logit-diff only (NN's own 1-d projection ceiling)
    diff_acc = float(((nn_diff > 0).astype(int) == labels).mean())

    out = {
        "n_events": int(len(labels)),
        "plane_ls_r2": r2,
        "nn_accuracy": nn_acc,
        "nn_logitdiff_accuracy": diff_acc,
        "plane_accuracy": plane_acc,
        "plane_auc": plane_auc,
        "plane_vs_nn_logitdiff_corr": corr,
    }
    print(out)
    (EXPERIMENTS / "esr" / "plane_ceiling.json").write_text(
        __import__("json").dumps(out, indent=2))


if __name__ == "__main__":
    main()
