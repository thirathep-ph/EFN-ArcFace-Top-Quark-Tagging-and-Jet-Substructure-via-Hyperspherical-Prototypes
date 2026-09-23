"""LorentzNet + ArcFace subclass analysis, compared to the EFN paper partition.

Runs the SAME pipeline the paper uses for EFN (spectral clustering + eigengap
on L2-normalized embeddings per TRUE class, max_k=5, largest cluster -> Core,
second -> Edge, subclass id = sc%2 + 2*cls) on the trained LorentzNet model,
then compares the discovered subclasses to the canonical EFN partition via ARI
and per-subclass DGLAP exponents P(zg) = A z^alpha (1-z)^beta.

Artifacts
---------
``experiments/lorentznet_embeddings.npz``  (full-test embeddings, cached)
``experiments/lorentznet_subclasses.json`` (k, sizes, ARI, DGLAP fits)
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
try:
    import torch_directml  # noqa: F401  (registers PrivateUse1 so checkpoint loads)
except ImportError:
    pass
from sklearn.metrics import adjusted_rand_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from arcefn.data.loader import load_awkward, JetTaggingDataset
from arcefn.models.lorentznet_arcface import LorentzNetArcFace
from arcefn.utils.clustering import spectral_clustering_subclass
from arcefn.utils.paths import DATA_DIR, EXPERIMENTS

from scripts.analysis.compute_dglap_fit import SUBCLASS_NAMES, fit_zg

DEFAULT_CHECKPOINT = EXPERIMENTS / "lorentznet_arcface" / "checkpoint.pt"
EFN_EMBEDDINGS = EXPERIMENTS / "robustness_s16_m05" / "embeddings.npz"
RF_CACHE = EXPERIMENTS / "rf_cache_test.npz"
MAX_K = 5


def build_model(device: torch.device, checkpoint: str = str(DEFAULT_CHECKPOINT),
                n_hidden: int = 72, n_layers: int = 6,
                c_weight: float = 1e-3, dropout: float = 0.2) -> LorentzNetArcFace:
    model = LorentzNetArcFace(
        s=16.0, m=0.5, embedding_dim=64, n_hidden=n_hidden,
        n_layers=n_layers, c_weight=c_weight, dropout=dropout,
    ).to(device)
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def compute_embeddings(device: torch.device, model, batch_size: int = 32,
                       chunk_size: int = 100000, max_events: int = 404000,
                       max_constits: int = 64,
                       cooldown_s: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Chunked forward pass over the full test set -> (embeddings, labels).

    cooldown_s: sleep seconds between chunks (reduces sustained GPU load to
    avoid thermal shutdown on the RX 5700 XT under DirectML).
    """
    total_len = min(max_events, 404000)
    emb_list, lab_list = [], []
    n_chunks = int(np.ceil(total_len / chunk_size))
    with torch.no_grad():
        for ci in range(n_chunks):
            start = ci * chunk_size
            n_ev = min(chunk_size, total_len - start)
            ev, lab, w = load_awkward(str(DATA_DIR / "test.h5"),
                                      max_events=n_ev, start_event=start, lazy=True)
            ds = JetTaggingDataset(ev, lab, w)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
            for x, y, w_b, m, _ in loader:
                x, m = x.to(device), m.to(device)
                if max_constits is not None:
                    x = x[:, :max_constits].contiguous()
                    m = m[:, :max_constits].contiguous()
                _, _, emb = model(x, mask=m)
                emb_list.append(emb.cpu().float().numpy())
                lab_list.append(y.cpu().numpy())
            del loader, ds, ev, lab, w
            gc.collect()
            print(f"  chunk {ci + 1}/{n_chunks} done ({time.strftime('%H:%M:%S')})")
            if cooldown_s > 0 and ci < n_chunks - 1:
                print(f"  cooldown {cooldown_s}s...")
                time.sleep(cooldown_s)
    return np.concatenate(emb_list), np.concatenate(lab_list)


def discover_subclasses(emb_norm: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Per true class: spectral clustering + largest->Core mapping."""
    out = np.full(len(labels), -1, dtype=int)
    for cls in (0, 1):
        mask = labels == cls
        sub, k = spectral_clustering_subclass(emb_norm[mask], max_k=MAX_K)
        counts = np.bincount(sub, minlength=int(sub.max()) + 1)
        order = np.argsort(-counts)  # largest -> Core (0)
        mapped = np.array([int(np.where(order == s)[0][0]) for s in sub])
        out[mask] = 2 * cls + mapped
        print(f"  class {cls}: k={k}, counts={counts.tolist()}")
    return out


def fit_all(zg: np.ndarray, sub: np.ndarray) -> dict:
    fits = {}
    for sc in range(4):
        sel = zg[sub == sc]
        r = fit_zg(sel)
        if r is None:
            fits[SUBCLASS_NAMES[sc]] = None
            continue
        fits[SUBCLASS_NAMES[sc]] = {
            "n": r["n"], "alpha": round(r["alpha"], 4), "beta": round(r["beta"], 4),
            "r2": round(r["r2"], 4), "A": round(r["A"], 4),
        }
    return fits


def physics_per_subclass(mass: np.ndarray, tau32: np.ndarray,
                         sub: np.ndarray) -> dict:
    """Per-subclass FastJet observables (mean jet mass, mean tau32)."""
    out = {}
    for sc in range(4):
        sel = sub == sc
        if sel.sum() == 0:
            out[SUBCLASS_NAMES[sc]] = None
            continue
        out[SUBCLASS_NAMES[sc]] = {
            "n": int(sel.sum()),
            "mass_mean_gev": round(float(mass[sel].mean()), 2),
            "tau32_mean": round(float(tau32[sel].mean()), 4),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="LorentzNet+ArcFace subclass POC")
    ap.add_argument("--max_test_events", type=int, default=404000)
    ap.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--out_tag", type=str, default="",
                    help="filename suffix (e.g. poc)")
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "dml"],
                    help="dml uses the AMD GPU (risk: thermal shutdown under "
                         "sustained load on this machine); cpu is safe/slower")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--chunk_size", type=int, default=100000)
    ap.add_argument("--cooldown", type=float, default=0.0,
                    help="seconds to sleep between chunks (GPU thermal safety)")
    args = ap.parse_args()

    max_events = min(args.max_test_events, 404000)
    suffix = args.out_tag or (f"_{max_events // 1000}k" if max_events < 404000 else "")
    out_emb = EXPERIMENTS / f"lorentznet_embeddings{suffix}.npz"
    out_json = EXPERIMENTS / f"lorentznet_subclasses{suffix}.json"

    t0 = time.time()
    device = torch.device("privateuseone:0" if args.device == "dml" else "cpu")
    print(f"Device: {device} ({args.device})")

    print("Building LorentzNet model + loading checkpoint...")
    model = build_model(device, args.checkpoint)

    if out_emb.exists():
        z = np.load(out_emb)
        ln_emb, ln_labels = z["embeddings"], z["labels"]
        print(f"Loaded cached embeddings {out_emb.name}")
    else:
        print(f"Computing embeddings on first {max_events} test events "
              f"(batch {args.batch_size}, chunk {args.chunk_size}, "
              f"cooldown {args.cooldown}s)...")
        ln_emb, ln_labels = compute_embeddings(
            device, model, batch_size=args.batch_size,
            chunk_size=args.chunk_size, max_events=max_events,
            cooldown_s=args.cooldown)
        np.savez_compressed(out_emb, embeddings=ln_emb, labels=ln_labels)
        print(f"Saved embeddings -> {out_emb}")

    ln_norm = ln_emb / (np.linalg.norm(ln_emb, axis=1, keepdims=True) + 1e-10)

    print("\n[1/4] LorentzNet subclass discovery (spectral, true labels)...")
    ln_sub = discover_subclasses(ln_norm, ln_labels)

    print("\n[2/4] EFN canonical partition (recomputed, same subset)...")
    ez = np.load(EFN_EMBEDDINGS)
    efn_emb, efn_labels = ez["embeddings"][:max_events], ez["labels"][:max_events]
    efn_norm = efn_emb / (np.linalg.norm(efn_emb, axis=1, keepdims=True) + 1e-10)
    efn_sub = discover_subclasses(efn_norm, efn_labels)

    assert np.array_equal(ln_labels, efn_labels), "label order mismatch"

    ari_global = adjusted_rand_score(efn_sub, ln_sub)
    ari_qcd = adjusted_rand_score(efn_sub[efn_labels == 0], ln_sub[ln_labels == 0])
    ari_top = adjusted_rand_score(efn_sub[efn_labels == 1], ln_sub[ln_labels == 1])

    print("\n[3/4] DGLAP fits per subclass (zg from FastJet cache)...")
    rf = np.load(RF_CACHE)
    feats = rf["features"][:max_events]
    zg, mass, tau32 = feats[:, 8], feats[:, 0], feats[:, 7]
    assert len(zg) == len(ln_sub)
    ln_fits = fit_all(zg, ln_sub)
    efn_fits = fit_all(zg, efn_sub)

    print("\n[4/4] Per-subclass FastJet physics (mass, tau32)...")
    ln_phys = physics_per_subclass(mass, tau32, ln_sub)
    efn_phys = physics_per_subclass(mass, tau32, efn_sub)

    out = {
        "model": "LorentzNet + ArcFace",
        "n_events": int(len(ln_labels)),
        "max_constits": 64,
        "checkpoint": str(args.checkpoint),
        "ari_vs_efn": {
            "global_4way": round(float(ari_global), 4),
            "qcd": round(float(ari_qcd), 4),
            "top": round(float(ari_top), 4),
        },
        "ln_subclass_sizes": {SUBCLASS_NAMES[sc]: int((ln_sub == sc).sum()) for sc in range(4)},
        "efn_subclass_sizes": {SUBCLASS_NAMES[sc]: int((efn_sub == sc).sum()) for sc in range(4)},
        "ln_dglap": ln_fits,
        "efn_dglap": efn_fits,
        "ln_physics": ln_phys,
        "efn_physics": efn_phys,
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    out_json.write_text(json.dumps(out, indent=2))
    print(f"\n=== RESULTS (saved {out_json}) ===")
    print(f"ARI vs EFN: global={ari_global:.4f}  QCD={ari_qcd:.4f}  Top={ari_top:.4f}")
    print(f"LN sizes:  {out['ln_subclass_sizes']}")
    print(f"EFN sizes: {out['efn_subclass_sizes']}")
    print("LN DGLAP:")
    for name, f in ln_fits.items():
        print(f"  {name:10s} {f}")
    print("EFN DGLAP:")
    for name, f in efn_fits.items():
        print(f"  {name:10s} {f}")
    print("LN FastJet physics:")
    for name, p in ln_phys.items():
        print(f"  {name:10s} {p}")
    print("EFN FastJet physics:")
    for name, p in efn_phys.items():
        print(f"  {name:10s} {p}")
    print(f"[COMPLETE] ({out['elapsed_seconds']}s)")


if __name__ == "__main__":
    main()