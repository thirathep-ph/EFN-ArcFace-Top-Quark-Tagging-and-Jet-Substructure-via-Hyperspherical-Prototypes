"""POC: validate symbolic distillation of EFN+ArcFace on the full 404k test set.

Pipeline (all closed form, no neural weights):
  theta_i = (d_eta_i, d_phi_i)          deterministic jet-axis-relative coords
  z_i     = E_i / Sum E                 energy weight
  t_i     = phi_sym(theta_i)            PySR equations (per-particle, 2 outputs)
  (U, V)  = Sum_i z_i * t_i             pooling (fixed arithmetic)
  score   = s * (V - U)                 decision: sigmoid(s(V-U)) = softmax(s cos)[Top]
  logit_G = rho_sym(U, V)               optional fitted readout (cross-check)

Metrics vs NN model + labels: acc, AUC, rej50, agreement with NN predictions
(on the 40k subset where NN logits were stored), and wall-time benchmark
(symbolic per-particle eval vs EFN phi MLP forward on the same batch).

Artifacts: experiments/esr/validation_results.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from arcefn.data.loader import load_awkward
from arcefn.models.efn_arcface import TopTaggingModel
from arcefn.utils.device import get_device
from arcefn.utils.paths import CHECKPOINT, DATA_DIR, EXPERIMENTS

S = 16.0
CHUNK = 50_000
BATCH = 2048


from typing import Callable


def compile_eq(eq: str) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Translate a PySR equation string into a numpy-vectorized callable (x0,x1)."""
    src = eq
    for name in ["square", "sqrt", "abs", "sin", "cos", "tanh"]:
        src = src.replace(f"{name}(", f"np.{name}(")
    src = src.replace("exp(", "safe_exp(").replace("log(", "safe_log(")
    ns = {"np": np, "safe_exp": safe_exp, "safe_log": safe_log}

    def f(x0, x1):
        ns["x0"], ns["x1"] = x0, x1
        return np.nan_to_num(eval(src, ns), nan=0.0, posinf=0.0, neginf=0.0)

    return f


def safe_exp(x):
    return np.exp(np.clip(x, -88.0, 88.0))


def safe_log(x):
    return np.log(np.clip(x, 1e-300, None))


def load_eq(path: str | Path, output: int | None = None) -> dict:
    d = json.load(open(path))
    if output is not None:
        front = d["equations_per_output"][output]
    else:
        front = d["equations"]
    return front[-1]  # best loss on the Pareto front


def main() -> None:
    phi0 = load_eq(EXPERIMENTS / "esr" / "phi_sym.json", output=0)
    phi1 = load_eq(EXPERIMENTS / "esr" / "phi_sym.json", output=1)
    rho = load_eq(EXPERIMENTS / "esr" / "readout_sym.json")
    f_t1 = compile_eq(phi0["equation"])
    f_t2 = compile_eq(phi1["equation"])
    f_G = compile_eq(rho["equation"])
    print(f"[esr-validate] phi_sym t1: {phi0['equation']}")
    print(f"[esr-validate] phi_sym t2: {phi1['equation']}")
    print(f"[esr-validate] rho_sym G:  {rho['equation']}")

    total = 404_000
    scores_all = np.empty(total, dtype=np.float64)
    logits_g_all = np.empty(total, dtype=np.float64)
    labels_all = np.empty(total, dtype=np.int32)
    t_eval = 0.0
    n_events_done = 0

    for c0 in range(0, total, CHUNK):
        c1 = min(c0 + CHUNK, total)
        events, labels, _ = load_awkward(str(DATA_DIR / "test.h5"), start_event=c0,
                                         max_events=c1 - c0, lazy=True)
        E = np.asarray(events.E, dtype=np.float32)
        px = np.asarray(events.px, dtype=np.float32)
        py = np.asarray(events.py, dtype=np.float32)
        pz = np.asarray(events.pz, dtype=np.float32)
        mask = (E > 0).astype(np.float32)

        pt = np.sqrt(px**2 + py**2 + 1e-10)
        eta = np.arcsinh(pz / (pt + 1e-10))
        phi = np.arctan2(py, px)
        jet_p4 = (np.stack([E, px, py, pz], axis=-1) * mask[..., None]).sum(axis=1)
        j_pt = np.sqrt(jet_p4[:, 1] ** 2 + jet_p4[:, 2] ** 2 + 1e-10)
        j_eta = np.arcsinh(jet_p4[:, 3] / (j_pt + 1e-10))
        j_phi = np.arctan2(jet_p4[:, 2], jet_p4[:, 1])
        d_eta = eta - j_eta[:, None]
        d_phi = phi - j_phi[:, None]
        d_phi = (d_phi + np.pi) % (2 * np.pi) - np.pi
        z = E / (E.sum(axis=1, keepdims=True) + 1e-10)
        z = z * mask

        # symbolic per-particle evaluation (vectorized, time it)
        n = len(labels)
        flat_eta = d_eta.reshape(-1)
        flat_phi = d_phi.reshape(-1)
        t0 = time.time()
        t1v = f_t1(flat_eta, flat_phi).reshape(n, -1)
        t2v = f_t2(flat_eta, flat_phi).reshape(n, -1)
        t_eval += time.time() - t0
        U = (t1v * z).sum(axis=1)
        V = (t2v * z).sum(axis=1)
        score = S * (V - U)
        logit_g = f_G(U, V)

        scores_all[c0:c1] = score
        logits_g_all[c0:c1] = logit_g
        labels_all[c0:c1] = labels.astype(np.int32)
        n_events_done += n
        print(f"[esr-validate] chunk {c0}-{c1} done ({n_events_done}/{total})")

    labels = labels_all
    pred = (scores_all > 0).astype(int)
    acc = float((pred == labels).mean())
    auc = float(roc_auc_score(labels, scores_all))
    fpr, tpr, _ = roc_curve(labels, scores_all)
    rej50 = 1.0 / float(fpr[np.argmin(np.abs(tpr - 0.5))]) if np.any(tpr >= 0.5) else None

    pred_g = (logits_g_all > 0).astype(int)
    acc_g = float((pred_g == labels).mean())
    auc_g = float(roc_auc_score(labels, logits_g_all)) if len(np.unique(labels)) > 1 else None

    # agreement with NN model on the 40k esr subset (NN logits stored there)
    d40 = np.load(EXPERIMENTS / "esr" / "jet.npz")
    nn_logit = d40["logit"].astype(np.float64)
    nn_pred = (nn_logit > 0).astype(int)
    # symbolic scores for exactly those 40k events: rerun the 40k subset
    events, labels40, _ = load_awkward(str(DATA_DIR / "test.h5"), max_events=40_000, lazy=True)
    rng = np.random.default_rng(42)
    idx = np.sort(rng.choice(40_000, 40_000, replace=False))  # same subset as esr targets
    labels40 = labels40[idx]
    E = np.asarray(events.E, dtype=np.float32)[idx]
    px = np.asarray(events.px, dtype=np.float32)[idx]
    py = np.asarray(events.py, dtype=np.float32)[idx]
    pz = np.asarray(events.pz, dtype=np.float32)[idx]
    mask = (E > 0).astype(np.float32)
    pt = np.sqrt(px**2 + py**2 + 1e-10)
    eta = np.arcsinh(pz / (pt + 1e-10))
    phi = np.arctan2(py, px)
    jet_p4 = (np.stack([E, px, py, pz], axis=-1) * mask[..., None]).sum(axis=1)
    j_pt = np.sqrt(jet_p4[:, 1] ** 2 + jet_p4[:, 2] ** 2 + 1e-10)
    j_eta = np.arcsinh(jet_p4[:, 3] / (j_pt + 1e-10))
    j_phi = np.arctan2(jet_p4[:, 2], jet_p4[:, 1])
    d_eta = eta - j_eta[:, None]
    d_phi = phi - j_phi[:, None]
    d_phi = (d_phi + np.pi) % (2 * np.pi) - np.pi
    z = E / (E.sum(axis=1, keepdims=True) + 1e-10)
    z = z * mask
    n40 = len(labels40)
    t1v = f_t1(d_eta.reshape(-1), d_phi.reshape(-1)).reshape(n40, -1)
    t2v = f_t2(d_eta.reshape(-1), d_phi.reshape(-1)).reshape(n40, -1)
    U = (t1v * z).sum(axis=1)
    V = (t2v * z).sum(axis=1)
    sym_score40 = S * (V - U)
    sym_pred40 = (sym_score40 > 0).astype(int)
    agreement = float((sym_pred40 == nn_pred).mean())

    # speed benchmark: symbolic vs NN phi on same batch of particles
    dev, _ = get_device()
    model = TopTaggingModel(s=16.0, m=0.5, embedding_dim=64, particle_dim=128)
    ck = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    model.load_state_dict(sd)
    model.to(dev)
    model.eval()
    xb = torch.from_numpy(np.stack([d_eta[:BATCH], d_phi[:BATCH]], -1)).to(dev)
    with torch.no_grad():
        t0 = time.time()
        for _ in range(5):
            _ = model.efn.phi(xb.reshape(-1, 2))
        t_nn = (time.time() - t0) / 5
    t0 = time.time()
    for _ in range(5):
        _ = f_t1(d_eta[:BATCH].reshape(-1), d_phi[:BATCH].reshape(-1))
        _ = f_t2(d_eta[:BATCH].reshape(-1), d_phi[:BATCH].reshape(-1))
    t_sym = (time.time() - t0) / 5

    out = {
        "n_events": int(n_events_done),
        "equations": {
            "phi_t1": phi0["equation"], "phi_t1_complexity": int(phi0["complexity"]),
            "phi_t2": phi1["equation"], "phi_t2_complexity": int(phi1["complexity"]),
            "readout": rho["equation"], "readout_complexity": int(rho["complexity"]),
        },
        "fit_loss": {"phi_t1": phi0["loss"], "phi_t2": phi1["loss"], "readout": rho["loss"]},
        "accuracy_plane": acc,
        "auc_plane": auc,
        "rejection_50tpr": rej50,
        "accuracy_readout": acc_g,
        "auc_readout": auc_g,
        "agreement_with_nn_40k": agreement,
        "speed_benchmark_per_particle": {
            "nn_phi_mlp_seconds": t_nn,
            "symbolic_seconds": t_sym,
            "speedup_x": float(t_nn / t_sym) if t_sym > 0 else None,
        },
        "total_symbolic_eval_seconds_404k": t_eval,
    }
    (EXPERIMENTS / "esr" / "validation_results.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
