"""POC: PySR symbolic distillation of EFN+ArcFace (replace weights with formulas).

Two-stage fit (data from compute_esr_targets.py, experiments/esr/):

  P1  per-particle:  (eta, phi) -> T (2D)   : symbolic phi_sym
  R1  jet level:     (U, V)     -> logit    : symbolic readout G (absorbs rho+cos+s)

POC budget: small niterations / populations so a run completes in ~30-60 min
on 4 cores. We store the best equation per complexity from the Pareto front.

Usage:
  python -m scripts.analysis.fit_esr --stage p1|r1 [--quick]

Artifacts (experiments/esr/):
  phi_sym.json / readout_sym.json  : equations + loss + complexity
  p1_fit.npz / r1_fit.npz          : predictions for validation
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


def load_p1() -> tuple[np.ndarray, np.ndarray]:
    d = np.load(EXPERIMENTS / "esr" / "perparticle.npz")
    return np.stack([d["eta"], d["phi"]], axis=1).astype(np.float64), d["T"].astype(np.float64)


def load_r1() -> tuple[np.ndarray, np.ndarray]:
    d = np.load(EXPERIMENTS / "esr" / "jet.npz")
    return np.stack([d["U"], d["V"]], axis=1).astype(np.float64), d["logit"].astype(np.float64)


def fit_p1(quick: bool, micro: bool, seed: int) -> None:
    X, T = load_p1()
    rng = np.random.default_rng(seed)
    n = 50_000 if micro else (200_000 if quick else 400_000)
    sel = rng.choice(len(X), n, replace=False)
    X, T = X[sel], T[sel]
    print(f"[p1] fitting phi_sym on {n} particles ...", flush=True)

    from pysr import PySRRegressor

    t0 = time.time()
    model = PySRRegressor(
        niterations=4 if micro else (15 if quick else 30),
        populations=4 if micro else (10 if quick else 20),
        population_size=30 if micro else 50,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["square", "sqrt", "abs", "sin", "cos", "exp"],
        maxsize=15 if micro else 25,
        maxdepth=6 if micro else 8,
        parsimony=1e-4,
        verbosity=0,
        progress=False,
        deterministic=False,
        parallelism="multithreading",
        procs=4,
        random_state=seed,
    )
    model.fit(X, T)
    eqs = model.equations_
    if isinstance(eqs, list):
        eqs_list = eqs  # multi-output: one Pareto front per target column
    else:
        eqs_list = [eqs]
    print(f"[p1] done in {time.time()-t0:.0f}s; {len(eqs_list)} output(s), "
          f"{len(eqs_list[0])} equations on Pareto front", flush=True)

    def _serialize(front) -> list[dict]:
        return [
            {
                "complexity": int(r["complexity"]),
                "loss": float(r["loss"]),
                "score": float(r["score"]) if r["score"] == r["score"] else None,
                "equation": str(r["equation"]),
            }
            for _, r in front.iterrows()
        ]

    out = {
        "stage": "p1",
        "n_samples": n,
        "fit_seconds": time.time() - t0,
        "n_outputs": len(eqs_list),
        "best_loss": [float(f["loss"].iloc[0]) for f in eqs_list],
        "equations_per_output": [_serialize(f) for f in eqs_list],
    }
    (EXPERIMENTS / "esr" / "phi_sym.json").write_text(json.dumps(out, indent=2))
    print(f"[p1] saved experiments/esr/phi_sym.json "
          f"(best losses {[float(f['loss'].iloc[0]) for f in eqs_list]})", flush=True)


def fit_r1(quick: bool, micro: bool, seed: int) -> None:
    X, Y = load_r1()
    rng = np.random.default_rng(seed + 7)
    n = 8_000 if micro else (15_000 if quick else 30_000)
    sel = rng.choice(len(X), n, replace=False)
    X, Y = X[sel], Y[sel]
    print(f"[r1] fitting readout G(U,V) on {n} jets ...", flush=True)

    from pysr import PySRRegressor

    t0 = time.time()
    model = PySRRegressor(
        niterations=4 if micro else (15 if quick else 30),
        populations=4 if micro else (10 if quick else 20),
        population_size=30 if micro else 50,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["square", "sqrt", "abs", "tanh", "exp", "log"],
        maxsize=18 if micro else 30,
        maxdepth=6 if micro else 10,
        parsimony=1e-4,
        verbosity=0,
        progress=False,
        deterministic=False,
        parallelism="multithreading",
        procs=4,
        random_state=seed,
    )
    model.fit(X, Y)
    eqs = model.equations_
    if isinstance(eqs, list):
        eqs = eqs[0]
    print(f"[r1] done in {time.time()-t0:.0f}s; {len(eqs)} equations on Pareto front", flush=True)
    out = {
        "stage": "r1",
        "n_samples": n,
        "fit_seconds": time.time() - t0,
        "best_loss": float(eqs["loss"].iloc[0]),
        "equations": [
            {
                "complexity": int(r["complexity"]),
                "loss": float(r["loss"]),
                "score": float(r["score"]) if r["score"] == r["score"] else None,
                "equation": str(r["equation"]),
            }
            for _, r in eqs.iterrows()
        ],
    }
    (EXPERIMENTS / "esr" / "readout_sym.json").write_text(json.dumps(out, indent=2))
    print(f"[r1] saved experiments/esr/readout_sym.json (best loss {out['best_loss']:.6f})", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["p1", "r1", "all"], default="all")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--micro", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.stage in ("p1", "all"):
        fit_p1(args.quick, args.micro, args.seed)
    if args.stage in ("r1", "all"):
        fit_r1(args.quick, args.micro, args.seed)


if __name__ == "__main__":
    main()
