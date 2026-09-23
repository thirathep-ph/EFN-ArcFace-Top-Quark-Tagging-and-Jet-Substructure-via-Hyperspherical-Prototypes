"""Traceability for the tau32-ablation claim in the symbolic-distillation section.

The paper states that disabling the tau32 term in the c=6 equation reduces
the score to tanh(m/pT) alone. This script makes that evaluation explicit and
reproducible: it takes the c=6 equation from fit_full.json, zeroes the tau32
input (the only place tau32 enters), and scores the resulting score function
end-to-end on the full 404k test set using the exact standardization block.

The c=6 equation is ``(Tau32 * -0.3476249) + tanh(Mass_over_pT)``, so zeroing
Tau32 yields ``tanh(Mass_over_pT)``, which is exactly the Pareto-front c=2
equation. The ablation accuracy therefore equals the c=2 accuracy evaluated
in `compute_symbolic_eval.py`.

Artifacts:
  experiments/physics_sr/symbolic_ablation.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from arcefn.utils.paths import EXPERIMENTS

OUT_DIR = EXPERIMENTS / "physics_sr"


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    acc = float(np.mean((y_pred > 0.5) == y_true))
    pos = np.sum(y_true == 1)
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

    import sympy as sp

    syms = {n: sp.Symbol(n) for n in names}
    eq_c6 = next(r["equation"] for r in fit["equations"] if r["complexity"] == 6)
    expr_full = sp.sympify(eq_c6, locals={"tanh": sp.tanh, **syms})

    # Zero out the tau32 input (the only place tau32 enters the c=6 equation).
    Xz_ablated = Xz.copy()
    Xz_ablated[:, names.index("Tau32")] = 0.0
    f = sp.lambdify(names, expr_full, "numpy")
    raw = np.asarray(f(*[Xz_ablated[:, i] for i in range(10)]), dtype=np.float64)
    y_hat = np.clip(raw, -50, 50) * y_std + y_mean

    # Cross-check: zeroing tau32 must reproduce the c=2 equation exactly.
    eq_c2 = next(r["equation"] for r in fit["equations"] if r["complexity"] == 2)
    expr_c2 = sp.sympify(eq_c2, locals={"tanh": sp.tanh, **syms})
    f2 = sp.lambdify(names, expr_c2, "numpy")
    y_hat_c2 = np.clip(np.asarray(f2(*[Xz[:, i] for i in range(10)]), dtype=np.float64),
                       -50, 50) * y_std + y_mean

    out = {
        "source": str(OUT_DIR / "fit_full.json"),
        "c6_equation": eq_c6,
        "c2_equation": eq_c2,
        "ablation": "Tau32 input zeroed in c=6 equation (only place tau32 enters)",
        "n": 404000,
        "ablated_metrics": _metrics(labels, y_hat),
        "c2_metrics": _metrics(labels, y_hat_c2),
        "max_abs_diff_ablated_vs_c2": float(np.max(np.abs(y_hat - y_hat_c2))),
        "paper_claim": "disabling tau32 reduces to tanh(m/pT) alone (86.6% acc, AUC 0.906)",
        "note": "Committed c=2 eval (symbolic_eval.json) gives acc 0.8824, AUC 0.9059; "
                "the paper's 86.6% was not traceable to a committed artifact and is "
                "replaced by the reproducible value here.",
    }
    (OUT_DIR / "symbolic_ablation.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()