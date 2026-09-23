"""pT smear 5% per-jet Gaussian — detector-like resolution proxy (404k)."""
import json, time
import numpy as np, torch
from torch.utils.data import DataLoader
from arcefn.utils.paths import DATA_DIR, EXPERIMENTS
from arcefn.utils.device import get_device
from arcefn.utils.model_loading import load_model_from_dir
from arcefn.data.loader import load_awkward, JetTaggingDataset
from pathlib import Path

def evaluate(model, events, labels, device, batch_size=2048):
    import numpy as np
    w = np.ones(len(labels))
    ds = JetTaggingDataset(events, labels, w)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    correct = total = 0
    with torch.no_grad():
        for x, y, w_, m, _ in dl:
            x, y, m = x.to(device), y.to(device), m.to(device)
            logits = model(x, mask=m)[0]
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / total

def smear_events(events, sigma=0.05, seed=42):
    rng = np.random.default_rng(seed)
    import awkward as ak
    # per-constituent independent Gaussian smear (detector resolution proxy)
    # scales shape matches jagged structure of events
    counts = ak.num(events)
    total = int(ak.sum(counts))
    scales_flat = rng.normal(1.0, sigma, size=total).clip(0.3, 1.7)
    scales = ak.unflatten(scales_flat, ak.to_list(counts))
    events_smeared = ak.zip({
        "E": events.E * scales,
        "px": events.px * scales,
        "py": events.py * scales,
        "pz": events.pz * scales,
    }, with_name="Momentum4D")
    return events_smeared

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="experiments/robustness_s16_m05")
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    device,_ = get_device()
    model = load_model_from_dir(Path(args.model_dir), device)
    from arcefn.utils.paths import DATA_DIR
    events, labels, weights = load_awkward(str(DATA_DIR/"test.h5"), max_events=404000, lazy=False)
    acc_clean = evaluate(model, events, labels, device)
    events_s = smear_events(events, sigma=args.sigma, seed=args.seed)
    acc_smeared = evaluate(model, events_s, labels, device)
    out = {"sigma": args.sigma, "seed": args.seed, "n_events": 404000,
           "accuracy_clean": float(acc_clean), "accuracy_smeared": float(acc_smeared),
           "delta_pp": float((acc_smeared-acc_clean)*100)}
    print(f"Clean {acc_clean:.4f} smeared {acc_smeared:.4f} delta {out['delta_pp']:.3f} pp")
    out_path = EXPERIMENTS / "pt_smear_results.json"
    # also per-model_dir copy
    json.dump(out, open(out_path,"w"), indent=2)
    json.dump(out, open(Path(args.model_dir)/"pt_smear_results.json","w"), indent=2)
    print(f"Saved {out_path} and {Path(args.model_dir)/'pt_smear_results.json'}")

if __name__ == "__main__":
    main()
