"""Invert a standardized PySR equation back to raw Buckingham-pi space.

Takes experiments/physics_sr/fit_<mode>.json (standardization block + equations)
and rewrites every Pareto-front equation in terms of the raw, physical
(dimensionless) variables, then re-evaluates the best equation on the same
sample to verify that metrics are invariant under the (affine) transform.

Usage:
  python -m scripts.analysis.compute_inverted_eval --fit fit_quick.json [--refit]

Output:
  experiments/physics_sr/inverted_<mode>.json: raw-space equations + metrics;
  with --refit also refits the best structure's constants in raw space via
  scipy curve_fit (SymbolFit-style constant refinement).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from arcefn.utils.paths import EXPERIMENTS

OUT_DIR = EXPERIMENTS / "physics_sr"
FEATURE_NAMES_RAW = ["Mass", "mSD", "Mult", "nSD", "sqrt_d12", "sqrt_d23",
                     "Tau21", "Tau32", "zg", "theta_g"]
FEATURE_NAMES_DIMLESS = ["Mass_over_pT", "mSD_over_pT", "Mult", "nSD",
                         "sqrt_d12", "sqrt_d23", "Tau21", "Tau32", "zg", "theta_g"]


def build_dimless_x() -> np.ndarray:
    feats = np.load(EXPERIMENTS / "rf_cache_test.npz")["features"].astype(np.float64)
    logits = np.load(OUT_DIR / "logits.npz")
    pt = np.maximum(logits["jet_pt"].astype(np.float64), 1e-3)
    mass = np.maximum(feats[:, 0], 0.0)
    m_sd = np.maximum(feats[:, 1], 0.0)
    return np.column_stack([mass / pt, m_sd / pt, feats[:, 2], feats[:, 3],
                            feats[:, 4], feats[:, 5], feats[:, 6], feats[:, 7],
                            feats[:, 8], feats[:, 9]])


def to_sympy(eq: str, names: dict[str, "sp.Symbol"]):
    import sympy as sp
    s = re.sub(r"square\(([^()]*)\)", r"(\1)**2", eq)
    s = s.replace("abs(", "Abs(")
    for op in ("tanh", "exp", "log", "sqrt", "Abs"):
        pass
    return sp.sympify(s, locals={"tanh": sp.tanh, "exp": sp.exp, "log": sp.log,
                                 "sqrt": sp.sqrt, "Abs": sp.Abs, **names})


def invert_equation(eq_str: str, std: dict | None, names: dict[str, "sp.Symbol"],
                    raw_symbols: dict[str, "sp.Symbol"]):
    """Return (expr_in_raw_space, latex_str)."""
    import sympy as sp
    expr = to_sympy(eq_str, names)
    if std is None:
        return expr, sp.latex(expr)
    norm = std.get("norm", "z")
    for i, f in enumerate(std["feature_names"]):
        raw = raw_symbols[f]
        if norm == "z":
            expr = expr.subs(names[f], (raw - std["x_mean"][i]) / std["x_std"][i])
        else:
            expr = expr.subs(names[f], raw / std["x_std"][i])
    expr = sp.simplify(expr)
    return expr, sp.latex(expr)


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default="fit_quick.json")
    ap.add_argument("--refit", action="store_true",
                    help="refine constants in raw space with curve_fit")
    args = ap.parse_args()

    import sympy as sp
    fit = json.loads((OUT_DIR / args.fit).read_text(encoding="utf-8"))
    std = fit["standardization"]
    feature_names = std["feature_names"] if std else FEATURE_NAMES_DIMLESS

    names = {f: sp.Symbol(f) for f in feature_names}
    raw_sym = {f: sp.Symbol(f) for f in feature_names}

    rows_out = []
    for r in fit["equations"]:
        expr_raw, latex = invert_equation(r["equation"], std, names, raw_sym)
        rows_out.append({"complexity": r["complexity"], "loss": r["loss"],
                         "equation_standardized": r["equation"],
                         "equation_raw": str(expr_raw), "latex": latex})

    X = build_dimless_x()
    data = np.load(OUT_DIR / f"fit_{fit['mode']}.npz")
    sel = data["sel"]
    y, labels = data["y"].astype(np.float64), data["labels"].astype(np.int64)
    Xs = X[sel]
    y_std = std["y_std"] if std else 1.0
    y_mean = std.get("y_mean", 0.0) if std else 0.0

    best_idx = int(fit["equations"][0]["complexity"]) and 0
    best_expr, _ = invert_equation(fit["best_equation"], std, names, raw_sym)
    f_raw = sp.lambdify(list(raw_sym.values()), best_expr, "numpy")
    y_hat = np.clip(f_raw(*[Xs[:, i] for i in range(Xs.shape[1])]), -50, 50) * y_std + y_mean

    from scipy.stats import pearsonr
    corr = float(pearsonr(y, y_hat)[0])
    met = _metrics(labels, y_hat)
    print(f"[invert] {args.fit}: corr={corr:.4f} (json {fit['corr_with_nn_logit']:.4f})")
    print(f"[invert] metrics: acc={met['acc']:.4f} auc={met['auc']:.4f} "
          f"rej50={met['rej50']:.2f} (json auc {fit['metrics_thresholded']['auc']:.4f})")
    print(f"[invert] raw-space best equation:\n  {best_expr}")

    out = {"fit": args.fit, "mode": fit["mode"], "n_samples": fit["n_samples"],
           "best_equation_raw": str(best_expr), "corr_with_nn_logit": corr,
           "metrics_thresholded": met,
           "json_corr": fit["corr_with_nn_logit"],
           "json_metrics": fit["metrics_thresholded"],
           "equations_raw": rows_out}

    if args.refit:
        refit = _refit_constants(best_expr, raw_sym, Xs, y)
        out["refit"] = refit
        print(f"[refit] loss {refit['loss_before']:.5f} -> {refit['loss_after']:.5f}")
        for name, before, after in refit["constants"]:
            d = abs(after - before) / max(abs(before), 1e-12)
            print(f"[refit]   {name}: {before:.6g} -> {after:.6g}  |d|={d*100:.2f}%")

    out_path = OUT_DIR / f"inverted_{fit['mode']}.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[invert] saved {out_path}")


def _refit_constants(expr: "sp.Expr", raw_sym: dict[str, "sp.Symbol"],
                     Xs: np.ndarray, y: np.ndarray) -> dict:
    """Replace every float literal in expr by a free constant and curve_fit."""
    import sympy as sp
    from scipy.optimize import curve_fit

    feature_syms = list(raw_sym.values())
    floats = sorted({a for a in sp.preorder_traversal(expr) if a.is_number and a.is_Float},
                    key=str)
    c_syms = [sp.Symbol(f"c{i}") for i in range(len(floats))]
    repl = {f: c for f, c in zip(floats, c_syms)}
    expr_c = expr.xreplace(repl)
    f_num = sp.lambdify(feature_syms + c_syms, expr_c, "numpy")
    p0 = np.array([float(f) for f in floats])

    def fmodel(X, *c):
        return np.clip(f_num(*[X[:, i] for i in range(Xs.shape[1])], *c), -50, 50)

    loss0 = float(np.mean((fmodel(Xs, *p0) - y) ** 2))
    n = min(len(Xs), 40000)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(Xs), n, replace=False)
    try:
        popt, _ = curve_fit(fmodel, Xs[idx], y[idx], p0=p0, maxfev=20000)
        succ = True
    except RuntimeError as exc:
        popt, succ = p0, False
        popt = p0
    y_after = fmodel(Xs, *popt)
    loss_after = float(np.mean((y_after - y) ** 2))
    return {"success": succ, "loss_before": loss0, "loss_after": loss_after,
            "n_fit": int(n),
            "constants": [(str(f), float(f), float(p)) for f, p in zip(floats, popt)],
            "equation_refit": str(expr_c)}


if __name__ == "__main__":
    main()
