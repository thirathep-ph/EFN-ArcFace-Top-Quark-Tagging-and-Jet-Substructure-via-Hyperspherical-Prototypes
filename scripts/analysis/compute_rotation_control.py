"""Random-rotation control for k*=2 (geometric artifact vs physics).

Question: does the eigengap k*=2 come from the data's intrinsic structure,
or merely from the hypersphere's orientation (prototypes sitting at
particular directions)? A global random rotation preserves ALL pairwise
distances/angles but moves the structure to arbitrary directions. If k*=2
and the partition survive rotation, the binary split is intrinsic to the
data, not an accident of orientation.

Method: random orthogonal Q (QR of Gaussian, 3 seeds) applied to L2-normalized
404k ArcFace embeddings; identical spectral+eigengap pipeline per predicted
class; ARI vs unrotated partition + k* recorded.

Run: python -m scripts.analysis.compute_rotation_control
Artifact: experiments/rotation_control.json
"""
import json
import numpy as np
from sklearn.metrics import adjusted_rand_score

from arcefn.utils.clustering import spectral_clustering_subclass
from arcefn.utils.paths import EXPERIMENTS
from arcefn.utils.pred_labels import arcface_preds_from_centers, check_cm


def random_orthogonal(d: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    q, _ = np.linalg.qr(rng.randn(d, d))
    return q


def main() -> None:
    npz = np.load(EXPERIMENTS / "robustness_s16_m05" / "embeddings.npz")
    E, y = npz["embeddings"], npz["labels"]
    assert len(E) == 404000
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-10)
    preds = arcface_preds_from_centers(E, EXPERIMENTS / "robustness_s16_m05" / "checkpoint.pt")
    check_cm(preds, y, "rotation_control")

    # reference partition (unrotated, predicted classes)
    ref = {}
    for cls in [0, 1]:
        mask = preds == cls
        sl, k = spectral_clustering_subclass(En[mask], max_k=5)
        ref[cls] = {"labels": sl, "k": int(k), "mask": mask}
        print(f"ref {'QCD' if cls == 0 else 'Top'}: k*={k}", flush=True)

    report = {"n_events": 404000, "partition": "predicted_labels_primary",
              "method": "global random orthogonal rotation, same pipeline per predicted class",
              "reference_k": {str(c): ref[c]["k"] for c in [0, 1]}, "seeds": {}}
    for seed in [42, 123, 7]:
        Q = random_orthogonal(En.shape[1], seed)
        Er = En @ Q
        # norms preserved by construction; assert once
        assert np.allclose(np.linalg.norm(Er, axis=1), 1.0, atol=1e-6)
        entry = {}
        for cls in [0, 1]:
            mask = ref[cls]["mask"]
            sl, k = spectral_clustering_subclass(Er[mask], max_k=5)
            ari = float(adjusted_rand_score(ref[cls]["labels"], sl))
            entry["QCD" if cls == 0 else "Top"] = {"k_star": int(k), "ari_vs_unref": round(ari, 3)}
            print(f"seed {seed} {'QCD' if cls == 0 else 'Top'}: k*={k} ARI={ari:.3f}", flush=True)
        report["seeds"][str(seed)] = entry

    out = EXPERIMENTS / "rotation_control.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
