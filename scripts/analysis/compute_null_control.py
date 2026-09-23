"""Null control: does k*=2 need real jet structure? (model x data table)

Exp 1 (trained model + noise data): uniform-direction constituents with
realistic multiplicity/energy scale, fed as (200,4) tensors straight to the
trained ArcFace model. If k*=2 balanced survives, ArcFace makes 2 clusters
out of anything (loss artifact). If not, k*=2 needs real data structure.

Exp 3 (untrained model + real data): fresh random-init TopTaggingModel,
forward only on real 50k test events. If k*=2 appears, architecture gives
it from birth; if not, training is necessary.

Predictions use each model's own centers, margin-free (same rule as paper).
Artifact: experiments/null_control.json
"""
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from arcefn.models.efn_arcface import TopTaggingModel
from arcefn.data.loader import load_awkward, JetTaggingDataset
from arcefn.utils.device import get_device
from arcefn.utils.paths import DATA_DIR, EXPERIMENTS
from arcefn.utils.model_loading import load_model_from_dir
from arcefn.utils.clustering import spectral_clustering_subclass
from arcefn.utils.pred_labels import normalize

N_NOISE = 20000
N_REAL = 50000
BATCH = 1024


def forward_all(model, loader, device):
    model.eval()
    embs = []
    with torch.no_grad():
        for x, y, w, m, _ in loader:
            out = model(x.to(device), mask=m.to(device))
            embs.append(out[-1].cpu().numpy())
    return np.vstack(embs)


def get_centers(model):
    c = model.arcface_head.class_centers.detach().cpu().numpy()
    return normalize(c)


def cluster_report(En, preds, tag):
    out = {"tag": tag}
    for cls in [0, 1]:
        mask = preds == cls
        n = int(mask.sum())
        if n < 100:
            out["pred_%d" % cls] = {"n": n, "k_star": None}
            print(f"  {tag} pred={cls}: n={n} (too few)", flush=True)
            continue
        sl, k = spectral_clustering_subclass(En[mask], max_k=5)
        from collections import Counter
        cnt = sorted(Counter(sl).values(), reverse=True)
        out["pred_%d" % cls] = {"n": n, "k_star": int(k), "counts": [int(c) for c in cnt[:2]]}
        print(f"  {tag} pred={cls}: n={n} k*={k} counts={cnt[:2]}", flush=True)
    return out


def main():
    device, _ = get_device()
    print(f"Device: {device}")
    report = {}

    # ---- reference statistics from one real batch (scale/multiplicity) ----
    events, labels, weights = load_awkward(str(DATA_DIR / "test.h5"), max_events=N_REAL, lazy=False)
    ds = JetTaggingDataset(events, labels, weights)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=0)
    probe_x, _, _, probe_m, _ = next(iter(loader))
    e_all = probe_x[..., 0]
    e_pos = e_all[e_all > 0]
    e_lo, e_hi = float(e_pos.min()), float(e_pos.quantile(0.99))
    mult = (probe_m > 0).sum(dim=1).numpy()
    print(f"noise calibration: E in [{e_lo:.2f}, {e_hi:.2f}], mult mean={mult.mean():.1f}")

    # ---- Exp 1: trained model + uniform noise ----
    print("Exp 1: trained ArcFace + noise")
    rng = np.random.RandomState(0)
    xn = np.zeros((N_NOISE, 200, 4), dtype=np.float32)
    mn = np.zeros((N_NOISE, 200), dtype=np.float32)
    for i in range(N_NOISE):
        m = int(rng.choice(mult))
        E_i = rng.uniform(e_lo, e_hi, size=m).astype(np.float32)
        dirs = rng.normal(size=(m, 3)).astype(np.float32)
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
        xn[i, :m, 0] = E_i
        xn[i, :m, 1:] = dirs * E_i[:, None]
        mn[i, :m] = 1.0
    nloader = DataLoader(TensorDataset(torch.from_numpy(xn), torch.from_numpy(mn)),
                         batch_size=BATCH, shuffle=False)
    model = load_model_from_dir(EXPERIMENTS / "robustness_s16_m05", device)
    model.eval()
    embs = []
    with torch.no_grad():
        for x, m in nloader:
            embs.append(model(x.to(device), mask=m.to(device))[-1].cpu().numpy())
    embs = np.vstack(embs)
    del model
    En = normalize(embs)
    centers = get_centers(load_model_from_dir(EXPERIMENTS / "robustness_s16_m05", "cpu"))
    preds = np.argmax(En @ centers.T, axis=1)
    print(f"  noise pred balance: {int((preds==0).sum())}/{int((preds==1).sum())}")
    report["exp1_trained_on_noise"] = cluster_report(En, preds, "exp1")

    # ---- Exp 3: untrained model + real data ----
    print("Exp 3: untrained net + real data")
    torch.manual_seed(0)
    fresh = TopTaggingModel(s=16.0, m=0.5, embedding_dim=64, particle_dim=128).to(device)
    fresh.eval()
    emb_r = forward_all(fresh, loader, device)
    del fresh
    Enr = normalize(emb_r)
    torch.manual_seed(0)
    fresh_c = TopTaggingModel(s=16.0, m=0.5, embedding_dim=64, particle_dim=128)
    rc = normalize(fresh_c.arcface_head.class_centers.detach().numpy())
    del fresh_c
    preds_r = np.argmax(Enr @ rc.T, axis=1)
    print(f"  untrained pred balance: {int((preds_r==0).sum())}/{int((preds_r==1).sum())}")
    report["exp3_untrained_on_real"] = cluster_report(Enr, preds_r, "exp3")

    out = EXPERIMENTS / "null_control.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
