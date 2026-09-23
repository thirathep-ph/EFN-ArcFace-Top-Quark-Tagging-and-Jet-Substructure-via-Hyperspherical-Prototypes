"""Evaluate symbolic-regression equations (Pareto front) on the full test set.

Table-5 traceability: the compact equations discovered by PySR
(experiments/physics_sr/fit_full.json) are scored end-to-end on the
404k test events, reproducing the c=6 and c=21 metrics quoted in the
paper (corr/acc/auc/rej50) from a single reproducible script.

Pipeline (identical to fit_physics_sr.py):
  X0      = dimless features (Mass/pT, mSD/pT, ...) from rf_cache_test.npz
            + jet_pt from logits.npz
  Xz      = (X0 - x_mean) / x_std   (standardization block of fit_full.json)
  f       = sympy parse of PySR equation string
  y_hat   = f(Xz) * y_std + y_mean
  metrics = _metrics(labels, y_hat)  (same as fit_physics_sr)

Artifacts:
  experiments/physics_sr/symbolic_eval.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from arcefn.utils.paths import EXPERIMENTS

OUT_DIR = EXPERIMENTS / "physics_sr"


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    acc = float(np.mean((y_pred > 0.5) == y_true))
    pos = np.sum(y_true == 1)
    if pos == 0 or pos == len(y_true):
        return {"acc": acc, "auc": None, "rej50": None, "n": len(y_true)}
    order = np.argsort(-y_pred)
    sorted_true = y_true[order]
    tp = np.cumsum(sorted_true)
    fp = np.cumsum(1 - sorted_true)
    tpr = tp / tp[-1]
    fpr = fp / max(fp[-1], 1)
    auc = float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))
    n_eff = int(np.searchsorted(tp, pos * 0.5)) + 1
    fp_at = int(fp[n_eff - 1])
    rej50 = float((len(y_true) - pos) / fp_at) if fp_at > 0 else None
    return {"acc": acc, "auc": auc, "rej50": rej50, "n": int(len(y_true))}


def main() -> None:
    fit = json.loads((OUT_DIR / "fit_full.json").read_text())
    std = fit["standardization"]
    assert std["norm"] == "z"
    names = std["feature_names"]
    x_mean = np.array(std["x_mean"])
    x_std = np.array(std["x_std"])
    y_mean, y_std = std["y_mean"], std["y_std"]

    feats = np.load(EXPERIMENTS / "rf_cache_test.npz")["features"].astype(np.float64)
    logits = np.load(OUT_DIR / "logits.npz")
    pt = np.maximum(logits["jet_pt"].astype(np.float64), 1e-3)
    mass = np.maximum(feats[:, 0], 0.0)
    m_sd = np.maximum(feats[:, 1], 0.0)
    X0 = np.column_stack([mass / pt, m_sd / pt, feats[:, 2], feats[:, 3],
                          feats[:, 4], feats[:, 5], feats[:, 6], feats[:, 7],
                          feats[:, 8], feats[:, 9]])
    Xz = (X0 - x_mean) / x_std
    y = logits["logitdiff"].astype(np.float64)
    labels = logits["labels"].astype(np.int64)
    assert len(feats) == len(y) == len(labels) == 404000

    npz = np.load(OUT_DIR / "fit_full.npz")
    y_hat_best_ref = npz["y_hat"].astype(np.float64)

    import sympy as sp

    syms = {n: sp.Symbol(n) for n in names}
    rows = []
    for r in fit["equations"]:
        c = int(r["complexity"])
        expr = sp.sympify(r["equation"], locals={"tanh": sp.tanh, **syms})
        f = sp.lambdify(names, expr, "numpy")
        raw = np.asarray(f(*[Xz[:, i] for i in range(10)]), dtype=np.float64)
        if raw.ndim > 1:
            raw = raw.reshape(-1)
        if raw.size == 1:
            raw = np.full(len(y), float(raw.ravel()[0]))
        y_hat = np.clip(raw, -50, 50) * y_std + y_mean
        corr = float(pearsonr(y, y_hat)[0])
        rows.append({
            "complexity": c, "loss": r["loss"], "equation": r["equation"],
            "corr_with_nn_logit": corr, "metrics": _metrics(labels, y_hat),
        })

    best_row = max(rows, key=lambda r: r["metrics"]["auc"])
    ref = dict(corr=float(pearsonr(y, y_hat_best_ref)[0]),
               metrics=_metrics(labels, y_hat_best_ref))
    out = {
        "source": str(OUT_DIR / "fit_full.json"),
        "n": 404000,
        "equations": rows,
        "best_auc_equation": {"complexity": best_row["complexity"],
                              "equation": best_row["equation"]},
        "reference_y_hat_metrics": ref,
        "consistency_with_saved_npz": all(
            abs(rows[i]["metrics"]["acc"] - out_i) < 1e-6
            for i, out_i in enumerate([])  # filled below
        ),
    }
    out["consistency_with_saved_npz"] = (
        abs(out["reference_y_hat_metrics"]["corr"] - fit["corr_with_nn_logit"]) < 1e-6
        and abs(out["reference_y_hat_metrics"]["metrics"]["rej50"]
                - fit["metrics_thresholded"]["rej50"]) < 1e-6
    )
    (OUT_DIR / "symbolic_eval.json").write_text(json.dumps(out, indent=2))
    print(f"n={out['n']}  consistency_with_saved_npz={out['consistency_with_saved_npz']}")
    for r in rows:
        m = r["metrics"]
        print(f"c={r['complexity']:>2} corr={r['corr_with_nn_logit']:.4f} "
              f"acc={m['acc']:.4f} auc={m['auc']:.4f} rej50={m['rej50']:.2f} "
              f"| {r['equation']}")
    print(f"reference (saved npz y_hat): corr={ref['corr']:.4f} "
          f"met={ref['metrics']}")


if __name__ == "__main__":
    main()