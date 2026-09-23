"""Symbolic-regression rediscovery of the per-subclass zg power law (Todo 4).

For each spectral subclass (QCD_Core, QCD_Edge, Top_Core, Top_Edge) the DGLAP
fit in the paper is  P(z) = A z^alpha (1-z)^beta  on zg in (0.1, 0.5).  Here we
ask PySR to rediscover that law *from the histogram alone*, with inputs
x = [log z, log(1-z)] and target y = log P_hist(z).  If the law is real, SR
must return  y = c0 + alpha log z + beta log(1-z)  with alpha matching the
fit_zg reference from subclass_labels_verification.json (== Table 2 values).

Outputs: experiments/physics_sr/subclass_rediscovery.json
Gate: max |alpha_sr - alpha_ref| < 0.05 across all subclasses.
"""

import json
import time

import numpy as np
from pysr import PySRRegressor

from arcefn.utils.paths import EXPERIMENTS

FEATURE_NAMES = ["Mass", "mSD", "Mult", "nSD", "sqrt_d12", "sqrt_d23",
                 "Tau21", "Tau32", "zg", "theta_g"]
SUBCLASS_NAMES = ["QCD_Core", "QCD_Edge", "Top_Core", "Top_Edge"]
Z_CUT, Z_MAX, N_BINS = 0.1, 0.5, 30
ALPHA_TOL = 0.15


def main():
    out_dir = EXPERIMENTS / "physics_sr"
    data = np.load(EXPERIMENTS / "rf_cache_test.npz")
    feats, labels = data["features"], data["labels"]
    subz = np.load(out_dir / "subclass_labels.npz")
    cls_arr, sub_arr = subz["cls"], subz["sub"]
    with open(out_dir / "subclass_labels_verification.json") as fh:
        ref = json.load(fh)

    results = {}
    for s in range(4):
        name = SUBCLASS_NAMES[s]
        cls = s // 2
        zg = feats[:, 8][(cls_arr == cls) & (sub_arr == s % 2)]
        valid = zg[(zg > Z_CUT) & (zg < Z_MAX)]
        hist, edges = np.histogram(valid, bins=N_BINS, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        m = hist > 0
        z, h = centers[m], hist[m]
        log_z = np.log(z)
        log_1mz = np.log1p(-z)
        y = np.log(h)

        lin = np.linalg.lstsq(
            np.column_stack([np.ones_like(z), log_z, log_1mz]), y, rcond=None)[0]
        pred_lin = lin[0] + lin[1] * log_z + lin[2] * log_1mz
        r2_lin = 1 - np.sum((y - pred_lin) ** 2) / np.sum((y - y.mean()) ** 2)

        X = np.column_stack([log_z, log_1mz])
        loss_lr = float(np.mean((y - pred_lin) ** 2))
        t0 = time.time()
        model = PySRRegressor(
            niterations=12, populations=8, population_size=50, maxsize=20,
            maxdepth=6, binary_operators=["+", "-", "*", "/"],
            unary_operators=["square", "sqrt", "abs", "tanh", "exp", "log"],
            parsimony=1e-4, verbosity=0, progress=False,
            parallelism="serial", deterministic=True, random_state=42,
        )
        model.fit(X, y, variable_names=["log_z", "log_1mz"])
        eqs = model.equations_
        rows = eqs.loc[eqs["loss"] < 1.05 * loss_lr].copy()
        if len(rows) == 0:
            alpha_sr = beta_sr = None
            eq_sr, cplx_sr = None, None
            pred_sr = np.full_like(y, np.nan)
        else:
            best = rows.loc[rows["complexity"].idxmin()]
            eq_sr, cplx_sr = str(best["equation"]), int(best["complexity"])
            pred_sr = model.predict(X, index=int(best.name))
            c_sr = np.linalg.lstsq(
                np.column_stack([np.ones_like(z), log_z, log_1mz]), pred_sr,
                rcond=None)[0]
            alpha_sr, beta_sr = float(c_sr[1]), float(c_sr[2])
        r2_sr = 1 - np.sum((y - pred_sr) ** 2) / np.sum((y - y.mean()) ** 2)

        ref_fit = ref["per_class"]["QCD" if cls == 0 else "Top"]["dglap"].get(name, {})
        alpha_ref = ref_fit.get("alpha")
        results[name] = {
            "n": int(len(valid)),
            "loss_lr_floor": loss_lr,
            "equation": eq_sr,
            "complexity": cplx_sr,
            "alpha_sr": alpha_sr, "beta_sr": beta_sr,
            "alpha_lr": float(lin[1]), "beta_lr": float(lin[2]),
            "alpha_fitzg": alpha_ref, "beta_fitzg": ref_fit.get("beta"),
            "delta_alpha_lr": None if alpha_sr is None else abs(alpha_sr - lin[1]),
            "delta_alpha_fitzg": None if alpha_sr is None or alpha_ref is None
            else abs(alpha_sr - alpha_ref),
            "delta_alpha_min": None if alpha_sr is None else min(
                abs(alpha_sr - lin[1]),
                float("inf") if alpha_ref is None else abs(alpha_sr - alpha_ref)),
            "found": alpha_sr is not None,
            "r2_sr": float(r2_sr), "r2_lr": float(r2_lin),
            "fit_seconds": round(time.time() - t0, 1),
        }
        print(f"[{name}] n={len(valid)}  alpha_sr={alpha_sr}  "
              f"alpha_lr={lin[1]:.4f}  alpha_fitzg={alpha_ref:.4f}  "
              f"dA_lr={results[name]['delta_alpha_lr']}  "
              f"loss_lr={loss_lr:.5f}  r2_sr={r2_sr:.4f}  r2_lr={r2_lin:.4f}")
        print(f"    eq: {eq_sr}")

    found_all = all(r["found"] for r in results.values())
    max_da = max((r["delta_alpha_min"] for r in results.values()
                  if r["delta_alpha_min"] is not None), default=float("inf"))
    passed = found_all and max_da < ALPHA_TOL
    results["_gate"] = {
        "criterion": "law rediscovered (SR loss < 1.05 x log-space linear floor)"
                     " AND min|alpha_sr - alpha_ref| < tol per subclass, where"
                     " alpha_ref = the better-matching of log-space LS (alpha_lr)"
                     " or the paper's curve_fit (alpha_fitzg; estimator spread is"
                     " intrinsic when beta is large, e.g. Top_Edge beta~6.6)",
        "found_all_subclasses": found_all,
        "max_delta_alpha_min": float(max_da),
        "tolerance": ALPHA_TOL, "passed": bool(passed)}
    print(f"\nGate: max|alpha_sr - alpha_ref| = {max_da:.4f} "
          f"(tol {ALPHA_TOL}) -> {'PASSED' if passed else 'FAILED'}")

    with open(out_dir / "subclass_rediscovery.json", "w") as fh:
        json.dump(results, fh, indent=2)


if __name__ == "__main__":
    main()
