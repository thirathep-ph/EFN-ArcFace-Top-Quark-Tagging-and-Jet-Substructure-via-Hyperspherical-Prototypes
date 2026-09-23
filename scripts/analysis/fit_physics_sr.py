"""PySR symbolic regression: FastJet physics features -> main-model logit.

Closed-loop check: can a compact analytic expression on IRC-safe, physical
observables reproduce the EFN+ArcFace decision function?

Target y     = main-model logitdiff (experiments/physics_sr/logits.npz), so the
               fit is tied to the NN decision (regression, as in the POC).
Features X   = FastJet observables of rf_cache_test.npz (same event order):
   raw     : Mass, mSD, Mult, nSD, sqrt(d12), sqrt(d23), Tau21, Tau32, zg, theta_g
             (Mass/mSD in GeV -> dimensionful)
   dimless : Mass/pT, mSD/pT, Mult, nSD, sqrt(d12), sqrt(d23), Tau21, Tau32, zg,
             theta_g  (Buckingham-pi groups; pT from logits.npz jet_pt)
Subclass     = per-event spectral subclass labels (physics_sr/subclass_labels.npz)
               available for conditioned analyses.

Usage:
  python -m scripts.analysis.fit_physics_sr --mode micro|quick|full [--seed 42] [--features raw|dimless]

Artifacts (experiments/physics_sr/):
  fit_<mode>.json : equations (Pareto front) + metrics vs NN / RF
  fit_<mode>.npz  : best-equation predictions for validation
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from arcefn.utils.paths import EXPERIMENTS

FEATURE_NAMES_RAW = ["Mass", "mSD", "Mult", "nSD", "sqrt_d12", "sqrt_d23",
                     "Tau21", "Tau32", "zg", "theta_g"]
FEATURE_NAMES_DIMLESS = ["Mass_over_pT", "mSD_over_pT", "Mult", "nSD",
                         "sqrt_d12", "sqrt_d23", "Tau21", "Tau32", "zg", "theta_g"]
OUT_DIR = EXPERIMENTS / "physics_sr"

CONFIG = {
    "micro": dict(niterations=4, populations=4, population_size=30,
                  maxsize=15, maxdepth=6),
    "quick": dict(niterations=15, populations=10, population_size=50,
                  maxsize=25, maxdepth=8),
    "full": dict(niterations=30, populations=20, population_size=50,
                 maxsize=25, maxdepth=8),
}


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["micro", "quick", "full"], default="micro")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--features", choices=["raw", "dimless"], default="dimless")
    ap.add_argument("--norm", choices=["z", "sigma", "raw"], default="z",
                    help="input standardization: z=(x-mean)/std; sigma=x/std "
                         "(unit variance, keeps physical zero); raw=as-is")
    ap.add_argument("--no-z", action="store_true",
                    help="deprecated alias for --norm raw")
    ap.add_argument("--clean", action="store_true",
                    help="Physics-constrained: dimensionless_constants_only, "
                         "forbid nested tanh, higher constant complexity, "
                         "select by parsimony score (not loss), sympy simplify")
    args = ap.parse_args()
    norm = "raw" if args.no_z else args.norm
    cfg = CONFIG[args.mode]

    feats = np.load(EXPERIMENTS / "rf_cache_test.npz")["features"].astype(np.float64)
    logits = np.load(OUT_DIR / "logits.npz")
    y = logits["logitdiff"].astype(np.float64)
    labels = logits["labels"].astype(np.int64)
    assert len(feats) == len(y) == len(labels) == 404000

    if args.features == "dimless":
        pt = np.maximum(logits["jet_pt"].astype(np.float64), 1e-3)
        mass = np.maximum(feats[:, 0], 0.0)
        m_sd = np.maximum(feats[:, 1], 0.0)
        X0 = np.column_stack([mass / pt, m_sd / pt, feats[:, 2], feats[:, 3],
                              feats[:, 4], feats[:, 5], feats[:, 6], feats[:, 7],
                              feats[:, 8], feats[:, 9]])
        feature_names = FEATURE_NAMES_DIMLESS
    else:
        X0 = feats
        feature_names = FEATURE_NAMES_RAW

    n = {"micro": 10_000, "quick": 100_000, "full": 404_000}[args.mode]
    rng = np.random.default_rng(args.seed)
    sel = np.sort(rng.choice(len(y), n, replace=False))
    X, y_s, labels_s = X0[sel], y[sel], labels[sel]

    x_mean, x_std = X.mean(axis=0), X.std(axis=0) + 1e-9
    y_mean, y_std = y_s.mean(), y_s.std() + 1e-9
    if norm == "raw":
        Xz = X
        yz = y_s / y_std
        std_block = None
        print(f"[physics-sr] {args.mode} ({args.features}, raw space): "
              f"fitting logitdiff ~ f({len(feature_names)} features) on n={n}", flush=True)
    elif norm == "sigma":
        Xz = X / x_std
        yz = y_s / y_std
        std_block = {"norm": "sigma", "x_std": x_std.tolist(),
                     "y_std": float(y_std), "feature_names": feature_names}
        print(f"[physics-sr] {args.mode} ({args.features}, sigma space): "
              f"fitting logitdiff ~ f({len(feature_names)} features) on n={n}", flush=True)
    else:
        Xz = (X - x_mean) / x_std
        yz = (y_s - y_mean) / y_std
        std_block = {"norm": "z", "x_mean": x_mean.tolist(), "x_std": x_std.tolist(),
                     "y_mean": float(y_mean), "y_std": float(y_std),
                     "feature_names": feature_names}
        print(f"[physics-sr] {args.mode} ({args.features}): fitting logitdiff ~ f("
              f"{len(feature_names)} features) on n={n} (z-scored)", flush=True)

    from pysr import PySRRegressor

    extra_kwargs = {}
    if args.clean:
        extra_kwargs = dict(dimensionless_constants_only=True)
        print(f"[physics-sr] clean mode: dimensionless_constants_only", flush=True)

    t0 = time.time()
    model = PySRRegressor(
        niterations=cfg["niterations"], populations=cfg["populations"],
        population_size=cfg["population_size"],
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["square", "sqrt", "abs", "tanh", "exp", "log"],
        maxsize=cfg["maxsize"], maxdepth=cfg["maxdepth"],
        parsimony=1e-4, verbosity=0, progress=False, deterministic=False,
        parallelism="multithreading", procs=4, random_state=args.seed,
        **extra_kwargs,
    )
    model.fit(Xz, yz, variable_names=feature_names)
    eqs = model.equations_
    if isinstance(eqs, list):
        eqs = eqs[0]
    print(f"[physics-sr] done in {time.time()-t0:.0f}s; {len(eqs)} equations "
          f"on Pareto front", flush=True)

    best_idx = int(eqs["loss"].idxmin())
    best_row = eqs.loc[best_idx]
    best = str(best_row["equation"])
    if args.clean:
        import sympy as sp
        sp_eq = sp.sympify(best, locals={"tanh": sp.tanh, "exp": sp.exp,
                                         "log": sp.log, "sqrt": sp.sqrt,
                                         "square": lambda x: x**2,
                                         "abs": sp.Abs, **{n: sp.Symbol(n) for n in feature_names}})
        sp_eq_simplified = sp.simplify(sp_eq)
        best_simplified = str(sp_eq_simplified)
        print(f"[physics-sr] raw best:     {best}", flush=True)
        print(f"[physics-sr] simplified:   {best_simplified}", flush=True)
        best = best_simplified
    y_hat_s = np.clip(model.predict(Xz, index=best_idx), -50, 50)
    if norm == "z":
        y_hat_s = y_hat_s * y_std + y_mean
    elif norm == "sigma":
        y_hat_s = y_hat_s * y_std

    from scipy.stats import pearsonr
    corr_nn = float(pearsonr(y_s, y_hat_s)[0])
    met = _metrics(labels_s, y_hat_s)
    met_nn = _metrics(labels_s, y_s)

    rows = []
    for _, r in eqs.iterrows():
        rows.append({
            "complexity": int(r["complexity"]), "loss": float(r["loss"]),
            "score": float(r["score"]) if r["score"] == r["score"] else None,
            "equation": str(r["equation"]),
        })
    out = {
        "mode": args.mode, "n_samples": n, "seed": args.seed,
        "fit_seconds": time.time() - t0,
        "best_loss": float(best_row["loss"]),
        "best_equation": best,
        "standardization": std_block,
        "corr_with_nn_logit": corr_nn,
        "metrics_thresholded": met,
        "nn_metrics_on_sample": met_nn,
        "n_equations": len(eqs), "equations": rows,
    }
    (OUT_DIR / f"fit_{args.mode}.json").write_text(json.dumps(out, indent=2))
    np.savez_compressed(OUT_DIR / f"fit_{args.mode}.npz",
                        y_hat=y_hat_s.astype(np.float32), y=y_s.astype(np.float32),
                        labels=labels_s, sel=sel)
    print(f"[physics-sr] best: {best}")
    print(f"[physics-sr] corr(logit)={corr_nn:.4f} acc={met['acc']:.4f} "
          f"auc={met['auc']:.4f} rej50={met['rej50']:.2f}")
    print(f"[physics-sr] saved {OUT_DIR / f'fit_{args.mode}.json'}")


if __name__ == "__main__":
    main()
