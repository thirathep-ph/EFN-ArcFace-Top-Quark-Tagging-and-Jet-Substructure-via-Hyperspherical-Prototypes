"""
Paired Model Statistics on the Full Test Set
=============================================
Evaluates all three trained checkpoints on the full 404,000-event test set
in aligned chunks (identical events for every model, so comparisons are
strictly paired), then computes:

* per-model accuracy with 95% binomial CIs (normal approximation and
  Clopper-Pearson exact),
* Hanley-McNeil standard errors for ROC AUC,
* pairwise McNemar tests (continuity-corrected chi-square and exact
  binomial two-sided) on the discordant prediction pairs,
* two-proportion power: smallest accuracy difference detectable at 80%
  power with alpha=0.05 for the given sample size,
* per-group sample sizes needed to detect Cohen's d effect sizes at 80%
  power (two-sample t-test approximation).

Output
------
* ``experiments/paired_model_stats.npz`` â€” aligned preds/scores/labels
* ``experiments/paired_model_stats.json`` â€” all statistics
"""

import json
import time
import gc
import numpy as np
from scipy import stats as sps
from sklearn.metrics import roc_auc_score

from arcefn.utils.paths import EXPERIMENTS, DATA_DIR
from arcefn.utils.device import get_device
from scripts.analysis.compute_bootstrap_ci import load_model, get_model_output

TEST_H5 = str(DATA_DIR / 'test.h5')
RESULTS_PATH = str(EXPERIMENTS / 'paired_model_stats.json')
NPZ_PATH = str(EXPERIMENTS / 'paired_model_stats.npz')

MODEL_CONFIGS = {
    "arcface_s16_m05": EXPERIMENTS / "robustness_s16_m05" / "checkpoint.pt",
    "linear_seed42": EXPERIMENTS / "baselines_linear_seed_42" / "checkpoint.pt",
    "coslinear_seed42": EXPERIMENTS / "baselines_coslinear_seed_42" / "checkpoint.pt",
}

Z_ALPHA_2 = 1.959964  # z_{0.025}
Z_BETA_80 = 0.841621  # z_{0.20}


def binomial_ci_normal(p: float, n: int) -> tuple[float, float]:
    se = np.sqrt(p * (1 - p) / n)
    return p - Z_ALPHA_2 * se, p + Z_ALPHA_2 * se


def binomial_ci_clopper_pearson(k: int, n: int) -> tuple[float, float]:
    res = sps.binomtest(k, n)
    return res.proportion_ci(confidence_level=0.95, method='exact')


def mcnemar(pred_a: np.ndarray, pred_b: np.ndarray, labels: np.ndarray):
    """McNemar test on aligned predictions vs. shared labels."""
    corr_a = pred_a == labels
    corr_b = pred_b == labels
    b = int(np.sum(corr_a & ~corr_b))  # A right, B wrong
    c = int(np.sum(~corr_a & corr_b))  # A wrong, B right
    n_disc = b + c
    chi2_cc = (abs(b - c) - 1.0) ** 2 / n_disc if n_disc > 0 else 0.0
    p_chi2 = sps.chi2.sf(chi2_cc, 1)
    m = min(b, c)
    p_exact = sps.binomtest(m, n_disc).pvalue if n_disc > 0 else 1.0
    return {"b": b, "c": c, "n_discordant": n_disc,
            "chi2_cc": chi2_cc, "p_chi2": p_chi2, "p_exact_binomial": p_exact}


def hanley_mcneil_se(auc_val: float, n_pos: int, n_neg: int) -> float:
    q1 = auc_val / (2.0 - auc_val)
    q2 = 2.0 * auc_val ** 2 / (1.0 + auc_val)
    se2 = (auc_val * (1.0 - auc_val)
           + (n_pos - 1) * (q1 - auc_val ** 2)
           + (n_neg - 1) * (q2 - auc_val ** 2)) / (n_pos * n_neg)
    return np.sqrt(max(se2, 0.0))


def main():
    device, _ = get_device()
    print(f"Device: {device}\n")

    all_preds, all_scores, all_labels = {}, {}, None

    for name, path in MODEL_CONFIGS.items():
        if not path.exists():
            print(f"  [SKIP] {name}")
            continue
        print(f"  [LOAD] {name}...", end=" ", flush=True)
        t0 = time.time()
        model = load_model(str(path), device)
        probs, preds, labels, _ = get_model_output(
            model, TEST_H5, device, max_events=1_000_000_000, seed=42)
        all_preds[name] = preds.astype(np.int8)
        all_scores[name] = probs[:, 1].astype(np.float32)
        if all_labels is None:
            all_labels = labels.astype(np.int8)
        else:
            assert np.array_equal(all_labels, labels), "label ordering mismatch"
        print(f"n={len(labels)} ({time.time()-t0:.0f}s)")
        del model
        gc.collect()

    n = len(all_labels)
    names = list(all_preds.keys())

    np.savez(NPZ_PATH, labels=all_labels,
             **{f"preds_{k}": v for k, v in all_preds.items()},
             **{f"scores_{k}": v for k, v in all_scores.items()})
    print(f"  Saved: {NPZ_PATH}")

    per_model = {}
    for name in names:
        acc = float(np.mean(all_preds[name] == all_labels))
        k = int(np.sum(all_preds[name] == all_labels))
        lo_n, hi_n = binomial_ci_normal(acc, n)
        lo_cp, hi_cp = binomial_ci_clopper_pearson(k, n)
        auc_val = float(roc_auc_score(all_labels, all_scores[name]))
        n_pos = int(all_labels.sum())
        hm_se = hanley_mcneil_se(auc_val, n_pos, n - n_pos)
        per_model[name] = {
            "n_events": n,
            "accuracy": acc,
            "n_correct": k,
            "ci95_normal": [lo_n, hi_n],
            "ci95_clopper_pearson": [lo_cp, hi_cp],
            "roc_auc": auc_val,
            "hanley_mcneil_se_auc": hm_se,
        }
        print(f"  {name}: acc={acc:.4f} "
              f"CI95_normal=[{lo_n:.4f},{hi_n:.4f}] "
              f"CI95_cp=[{lo_cp:.4f},{hi_cp:.4f}] "
              f"AUC={auc_val:.4f} SE={hm_se:.4f}")

    pbar = np.mean([per_model[m]["accuracy"] for m in names])
    two_prop = Z_ALPHA_2 + Z_BETA_80
    d_min = two_prop * np.sqrt(2 * pbar * (1 - pbar) / n)
    power = {
        "n_events": n,
        "p_bar": pbar,
        "detectable_diff_80pct_two_proportion": d_min,
        "per_group_n_for_cohens_d": {
            "d=0.2": 2 * two_prop ** 2 / 0.2 ** 2,
            "d=0.5": 2 * two_prop ** 2 / 0.5 ** 2,
            "d=0.8": 2 * two_prop ** 2 / 0.8 ** 2,
            "d=1.73": 2 * two_prop ** 2 / 1.73 ** 2,
        },
        "subclass_counts_from_interpretation": {
            "qcd_core": 123322, "qcd_edge": 78592,
            "top_core": 101260, "top_edge": 100826,
        },
    }
    print(f"  Detectable diff @80% power: {d_min*100:.2f} pp")

    mcnemar_results = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            key = f"{names[i]}_vs_{names[j]}"
            mcnemar_results[key] = mcnemar(all_preds[names[i]], all_preds[names[j]], all_labels)
            r = mcnemar_results[key]
            print(f"  McNemar {key}: b={r['b']} c={r['c']} "
                  f"chi2_cc={r['chi2_cc']:.1f} p={r['p_chi2']:.2e} "
                  f"(exact p={r['p_exact_binomial']:.2e})")

    results = {
        "config": {"n_events": n, "models": names, "seed": 42},
        "per_model": per_model,
        "mcnemar_pairwise": mcnemar_results,
        "power": power,
    }
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {RESULTS_PATH}")


if __name__ == '__main__':
    main()
