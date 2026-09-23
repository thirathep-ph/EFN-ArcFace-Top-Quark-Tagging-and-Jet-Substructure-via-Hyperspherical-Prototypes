"""Compute classification accuracy of a simple jet-mass cut on the full test set.

Backs the appendix claim that a mass cut at 160 GeV achieves 83.8% accuracy
and the optimal cut near 140 GeV reaches 88.9% (well below the model's 91.84%).

Uses the cached RF feature array (experiments/rf_cache_test.npz) which holds
the 10 physics features for all 404,000 test events (column 0 = jet mass).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from arcefn.utils.paths import EXPERIMENTS

CACHE = Path(EXPERIMENTS) / "rf_cache_test.npz"
OUT = Path(EXPERIMENTS) / "mass_cut_results.json"
CUTS = [100.0, 120.0, 140.0, 150.0, 160.0, 172.0]


def main() -> None:
    data = np.load(CACHE)
    masses = data["features"][:, 0]
    labels = data["labels"]

    results = {}
    for cut in CUTS:
        preds = (masses >= cut).astype(int)
        acc = float((preds == labels).mean())
        results[f"{int(cut)}_GeV"] = {"accuracy": acc}
        print(f"mass cut >= {cut:.0f} GeV: accuracy = {acc:.4f}")

    best_cut = max(CUTS, key=lambda c: results[f"{int(c)}_GeV"]["accuracy"])
    summary = {
        "n_events": int(len(labels)),
        "cuts_ge": results,
        "best_cut_GeV": int(best_cut),
        "best_accuracy": results[f"{int(best_cut)}_GeV"]["accuracy"],
        "note": (
            "Simple mass-only classifier (predict Top if mass >= cut, else QCD). "
            "Best cut ~140 GeV reaches 88.9%, below the EFN+ArcFace model's 91.84%."
        ),
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
