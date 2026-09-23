"""Save per-event spectral subclass labels for the 404k test set.

Replicates the exact clustering pipeline of
``compute_subclass_clustering.py`` (spectral + eigengap, cosine affinity,
largest cluster -> Core) but stores the per-event assignment so later
symbolic-regression stages can condition on subclasses directly.

The script self-verifies against the published aggregates
(``experiments/subclass_clustering_comparison.json``): cluster counts and
per-subclass DGLAP fits must reproduce.

Run: python -m scripts.analysis.compute_subclass_labels
Artifacts (experiments/physics_sr/):
  subclass_labels.npz : sub (404000,) int8  (0=Core, 1=Edge), cls (404000,) int8
  subclass_labels_verification.json
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from arcefn.utils.clustering import spectral_clustering_subclass
from arcefn.utils.paths import EXPERIMENTS
from scripts.analysis.compute_dglap_fit import SUBCLASS_NAMES, fit_zg

EMBEDDINGS = EXPERIMENTS / "robustness_s16_m05" / "embeddings.npz"
FEATURES = EXPERIMENTS / "rf_cache_test.npz"
REFERENCE = EXPERIMENTS / "subclass_clustering_comparison.json"
OUT_DIR = EXPERIMENTS / "physics_sr"


def main() -> None:
    t0 = time.time()
    emb = np.load(EMBEDDINGS)
    embeddings, labels = emb["embeddings"], emb["labels"]
    feat = np.load(FEATURES)
    zg_all = feat["features"][:, 8]
    assert len(embeddings) == len(labels) == len(zg_all) == 404000

    ref = json.loads(REFERENCE.read_text())["methods"]

    sub = np.full(len(labels), -1, dtype=np.int8)
    report = {"n_events": int(len(labels)), "per_class": {}}
    for cls in (0, 1):
        cls_name = "QCD" if cls == 0 else "Top"
        x = embeddings[labels == cls]
        x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-10)
        sc_labels, k_sc = spectral_clustering_subclass(x, max_k=5)
        counts = Counter(sc_labels)
        order = sorted(counts.keys(), key=lambda c: -counts[c])
        mapped = np.array([order.index(l) if l in order else 0 for l in sc_labels])
        sub[labels == cls] = mapped

        entry = {"k": int(k_sc), "counts": {}, "dglap": {}}
        for idx, c in enumerate(order[:2]):
            name = SUBCLASS_NAMES[2 * cls + idx]
            n_c = int(np.sum(mapped == idx))
            entry["counts"][name] = n_c
            fit = fit_zg(zg_all[labels == cls][mapped == idx])
            if fit:
                entry["dglap"][name] = {
                    "alpha": fit["alpha"], "beta": fit["beta"],
                    "r2": fit["r2"], "n": fit["n"],
                }
        report["per_class"][cls_name] = entry

        ref_ns = ref[cls_name]["partitions"]["spectral_eigengap"]
        ref_dg = ref_ns["dglap"]
        print(f"[{cls_name}] k={k_sc} (ref k={ref_ns['k']})")
        for name in entry["counts"]:
            ours_n = entry["counts"][name]
            ref_n = ref_ns["n0"] if "Core" in name else ref_ns["n1"]
            a_ours = entry["dglap"].get(name, {}).get("alpha")
            a_ref = ref_dg.get(name, {}).get("alpha")
            ok_n = ours_n == ref_n
            ok_a = a_ours is not None and a_ref is not None and abs(a_ours - a_ref) < 1e-3
            print(f"  {name:10s} n={ours_n} (ref {ref_n}) {'OK' if ok_n else 'MISMATCH'} | "
                  f"alpha={a_ours:.4f} (ref {a_ref:.4f}) {'OK' if ok_a else 'MISMATCH'}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_DIR / "subclass_labels.npz", sub=sub, cls=labels)
    report["elapsed_seconds"] = round(time.time() - t0, 1)
    (OUT_DIR / "subclass_labels_verification.json").write_text(
        json.dumps(report, indent=2)
    )
    print(f"[subclass-labels] saved {OUT_DIR / 'subclass_labels.npz'} "
          f"in {report['elapsed_seconds']}s")


if __name__ == "__main__":
    main()
