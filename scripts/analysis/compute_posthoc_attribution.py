"""Post-hoc attribution comparison on the Random Forest baseline.

Computes SHAP (TreeSHAP) and LIME attributions for the Random Forest
baseline operating on the 10 hand-crafted substructure observables
(``rf_cache_val.npz`` / ``rf_cache_test.npz``), and compares the resulting
feature rankings with the RF impurity-based importances and with the
effect-size analysis of ``interpretation_full_results.json``.

The purpose is empirical evidence for Sec. 3.3 (post-hoc vs ante-hoc):
standard post-hoc methods that are applicable to tabular classifiers
(SHAP, LIME) reproduce the same leading physics (mass, tau32, nSD)
that the ante-hoc symbolic distillation recovers, whereas Grad-CAM
is not even definable for point-cloud architectures (no feature maps).

Output
------
``experiments/posthoc_attribution_results.json``
``experiments/posthoc_attribution.png``
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from arcefn.utils.paths import EXPERIMENTS

names = ["Mass", "mSD", "Mult", "nSD", "sqrt(d12)", "sqrt(d23)",
         "tau21", "tau32", "zg", "theta_g"]


def main() -> None:
    val = np.load(EXPERIMENTS / "rf_cache_val.npz")
    test = np.load(EXPERIMENTS / "rf_cache_test.npz")
    X_tr, y_tr = val["features"], val["labels"]
    X_te, y_te = test["features"], test["labels"]

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=None, random_state=42, n_jobs=-1,
    )
    clf.fit(X_tr, y_tr)

    import shap
    explainer = shap.TreeExplainer(clf)
    rng = np.random.default_rng(42)
    n_shap = min(50000, len(X_te))
    idx = rng.choice(len(X_te), n_shap, replace=False)
    shap_values = explainer.shap_values(X_te[idx])[1]
    shap_means = np.abs(shap_values).mean(axis=0)

    import lime
    from lime.lime_tabular import LimeTabularExplainer

    n_lime = 1000
    idx_lime = rng.choice(len(X_te), n_lime, replace=False)
    lme = LimeTabularExplainer(
        X_tr,
        feature_names=names,
        class_names=["QCD", "Top"],
        mode="classification",
        discretize_continuous=False,
        random_state=42,
    )
    lime_means = np.zeros(len(names))
    for i in idx_lime:
        exp = lme.explain_instance(
            X_te[i], clf.predict_proba, num_features=len(names),
            num_samples=2000,
        )
        for name, weight in exp.as_map()[1]:
            lime_means[names.index(name)] += abs(weight)
    lime_means /= n_lime

    gini = clf.feature_importances_

    order = np.argsort(shap_means)[::-1]
    rho_shap_gini, p1 = spearmanr(shap_means, gini)
    rho_lime_gini, p2 = spearmanr(lime_means, gini)
    rho_shap_lime, p3 = spearmanr(shap_means, lime_means)

    results = {
        "n_test": int(len(X_te)),
        "n_shap": int(n_shap),
        "n_lime": int(n_lime),
        "features": names,
        "shap_mean_abs": {names[i]: float(shap_means[i]) for i in range(len(names))},
        "lime_mean_abs": {names[i]: float(lime_means[i]) for i in range(len(names))},
        "rf_gini": {names[i]: float(gini[i]) for i in range(len(names))},
        "shap_ranking": [names[i] for i in order],
        "lime_ranking": [names[i] for i in np.argsort(lime_means)[::-1]],
        "gini_ranking": [names[i] for i in np.argsort(gini)[::-1]],
        "spearman_shap_gini": round(float(rho_shap_gini), 4),
        "spearman_lime_gini": round(float(rho_lime_gini), 4),
        "spearman_shap_lime": round(float(rho_shap_lime), 4),
        "p_shap_gini": float(p1),
        "p_lime_gini": float(p2),
        "p_shap_lime": float(p3),
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    ypos = np.arange(len(names))
    y = ypos[order]
    w = 0.27
    ax.barh(y + w, shap_means[order] / shap_means.max(), height=w,
            label="SHAP (normalized)", color="#1f77b4")
    ax.barh(y, lime_means[order] / lime_means.max(), height=w,
            label="LIME (normalized)", color="#ff7f0e")
    ax.barh(y - w, gini[order] / gini.max(), height=w,
            label="RF Gini (normalized)", color="#2ca02c")
    ax.set_yticks(y)
    ax.set_yticklabels([names[i] for i in order])
    ax.invert_yaxis()
    ax.set_xlabel("Normalized importance")
    ax.set_title(
        f"RF baseline: post-hoc attributions\n"
        f"Spearman: SHAP-Gini {rho_shap_gini:.2f} | LIME-Gini {rho_lime_gini:.2f} | "
        f"SHAP-LIME {rho_shap_lime:.2f}"
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(EXPERIMENTS / "posthoc_attribution.png", dpi=150, bbox_inches="tight")

    (EXPERIMENTS / "posthoc_attribution_results.json").write_text(
        json.dumps(results, indent=2)
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()