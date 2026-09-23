"""Dimension-ablation evaluation: single 404k test-set pass per model.

Evaluates the ``embedding_dim=2`` ArcFace model (trained with the canonical
config: s=16, m=0.5, particle_dim=128, seed 42, 50 epochs) against the
canonical 64-d reference on the full test set, and decides the equivalence
gate that drives the paper's dimensionality claim.

Artifacts
---------
``experiments/dim_ablation_results.json``
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import chi2 as chi2_dist
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from arcefn.data.loader import load_awkward, JetTaggingDataset
from arcefn.models.efn_arcface import TopTaggingModel
from arcefn.utils.device import get_device
from arcefn.utils.paths import DATA_DIR, EXPERIMENTS

MAX_EVENTS = 404000
BATCH_SIZE = 2048

ABLATION_DIR = EXPERIMENTS / "ablation_dim_2"
REF_DIR = EXPERIMENTS / "robustness_s16_m05"


def load_config(run_dir: Path) -> dict:
    """Read the training config written alongside the checkpoint, with sensible fallbacks."""
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        return {
            "embedding_dim": int(cfg.get("embedding_dim", 64)),
            "scale": float(cfg.get("scale", 16.0)),
            "margin": float(cfg.get("margin") or 0.5),
            "particle_dim": int(cfg.get("particle_dim", 128)),
        }
    name = run_dir.name
    emb = 2 if name == "ablation_dim_2" else 64
    return {"embedding_dim": emb, "scale": 16.0, "margin": 0.5, "particle_dim": 128}


def model_predictions(run_dir: Path, cfg: dict, device: torch.device):
    """Return (labels, preds, scores) for one model on the full test set.

    ``scores`` = softmax(logits)[:, 1] (P(Top)), the same convention used by
    ``compute_bootstrap_ci`` for the canonical AUC/rejection numbers.
    """
    ckpt = run_dir / "checkpoint.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"no checkpoint: {ckpt}")

    model = TopTaggingModel(
        s=cfg["scale"],
        m=0.0,
        embedding_dim=cfg["embedding_dim"],
        particle_dim=cfg["particle_dim"],
    ).to(device)
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    sd = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    model.load_state_dict(sd, strict=True)
    model.eval()

    events, labels, weights = load_awkward(
        str(DATA_DIR / "test.h5"), max_events=MAX_EVENTS, lazy=False
    )
    ds = JetTaggingDataset(events, labels, weights)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    logits_all, preds = [], []
    with torch.no_grad():
        for x, y, w, m, _ in loader:
            x, m = x.to(device), m.to(device)
            logits, _, _ = model(x, mask=m)
            logits_all.append(logits.cpu().numpy())
            preds.append(logits.argmax(dim=1).cpu().numpy())
    logits_all = np.concatenate(logits_all)
    preds = np.concatenate(preds)
    probs = torch.softmax(torch.from_numpy(logits_all), dim=1).numpy()
    return labels, preds, probs[:, 1].astype(np.float64)


def rejection_at_tpr(labels: np.ndarray, scores: np.ndarray, target: float = 0.5) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = int(np.argmin(np.abs(tpr - target)))
    return float(1.0 / fpr[idx]) if fpr[idx] > 0 else None


def mcnemar_cc(p1: np.ndarray, p2: np.ndarray, labels: np.ndarray) -> dict:
    """Continuity-corrected McNemar on discordant pairs (identical event order)."""
    b = int(np.sum((p1 == 1) & (p2 == 0) & (labels == labels)))  # d2 correct, ref wrong
    c = int(np.sum((p1 == 0) & (p2 == 1) & (labels == labels)))  # d2 wrong, ref correct
    chi2 = (abs(b - c) - 1.0) ** 2 / (b + c) if (b + c) > 0 else 0.0
    p = float(chi2_dist.sf(chi2, 1))
    return {"d2_correct_ref_wrong": b, "d2_wrong_ref_correct": c, "chi2_cc": chi2, "p": p}


def metrics(labels: np.ndarray, preds: np.ndarray, scores: np.ndarray) -> dict:
    n = len(labels)
    tp = int(np.sum((preds == 1) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    acc = (tp + tn) / n
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    auc = float(roc_auc_score(labels, scores))
    se = 1.96 * np.sqrt(acc * (1 - acc) / n)
    return {
        "accuracy": round(float(acc), 6),
        "accuracy_ci95": [round(acc - se, 6), round(acc + se, 6)],
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "precision": round(float(prec), 6),
        "recall": round(float(rec), 6),
        "f1": round(float(f1), 6),
        "roc_auc": round(auc, 6),
        "rejection_50tpr": round(rejection_at_tpr(labels, scores), 3),
        "n_events": n,
    }


def main() -> None:
    t0 = time.time()
    device, _ = get_device()

    cfg_abl = load_config(ABLATION_DIR)
    cfg_ref = load_config(REF_DIR)
    print(f"ablation model config: {cfg_abl}")
    print(f"reference model config: {cfg_ref}")

    labels, preds_abl, scores_abl = model_predictions(ABLATION_DIR, cfg_abl, device)
    print(f"ablation forward pass done ({time.time() - t0:.0f}s)")
    _, preds_ref, scores_ref = model_predictions(REF_DIR, cfg_ref, device)
    print(f"reference forward pass done ({time.time() - t0:.0f}s)")

    m_abl = metrics(labels, preds_abl, scores_abl)
    m_ref = metrics(labels, preds_ref, scores_ref)
    mc = mcnemar_cc(preds_abl, preds_ref, labels)

    delta_acc = m_abl["accuracy"] - m_ref["accuracy"]
    delta_auc = m_abl["roc_auc"] - m_ref["roc_auc"]
    rej_abl = m_abl["rejection_50tpr"]
    rej_ref = m_ref["rejection_50tpr"]

    equivalent = (
        abs(delta_acc) < 0.002
        and mc["p"] > 0.05
        and abs(delta_auc) < 0.003
        and abs(rej_abl - rej_ref) < 30.0
    )

    out = {
        "ablation": {"dir": str(ABLATION_DIR), "config": cfg_abl, **m_abl},
        "reference": {"dir": str(REF_DIR), "config": cfg_ref, **m_ref},
        "comparison": {
            "delta_accuracy": round(delta_acc, 6),
            "delta_auc": round(delta_auc, 6),
            "rejection_50tpr_diff": round(rej_abl - rej_ref, 3),
            "mcnemar": mc,
            "gate_criteria": {
                "abs_delta_acc_lt_0.2pp": abs(delta_acc) < 0.002,
                "mcnemar_p_gt_0.05": mc["p"] > 0.05,
                "abs_delta_auc_lt_0.003": abs(delta_auc) < 0.003,
                "rej50_diff_lt_30": abs(rej_abl - rej_ref) < 30.0,
            },
            "equivalent": equivalent,
        },
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    path = EXPERIMENTS / "dim_ablation_results.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {path}")
    print(f"  d=2  : acc={m_abl['accuracy']:.6f} auc={m_abl['roc_auc']:.4f} rej50={rej_abl}")
    print(f"  d=64 : acc={m_ref['accuracy']:.6f} auc={m_ref['roc_auc']:.4f} rej50={rej_ref}")
    print(f"  delta_acc={delta_acc:+.6f}  delta_auc={delta_auc:+.6f}")
    print(f"  McNemar chi2={mc['chi2_cc']:.3f} p={mc['p']:.3g}")
    print(f"  EQUIVALENT = {equivalent}")


if __name__ == "__main__":
    main()