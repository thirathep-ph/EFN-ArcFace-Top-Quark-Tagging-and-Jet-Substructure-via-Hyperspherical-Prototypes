"""Δη-Δφ planes per class and per discovered subclass, plus representative jets.

Average density (hist2d) per (sub)class + single representative jet (medoid: closest to subclass centroid).
Reuses spectral clustering + stratified sampling pattern from compute_subclass_lund_planes.py.
"""
from __future__ import annotations
import argparse, json, time, os
from collections import Counter
from shutil import copy2
import matplotlib
matplotlib.use("Agg")
import numpy as np
import awkward as ak
from arcefn.data.loader import load_events_by_indices
from arcefn.utils.clustering import spectral_clustering_subclass
from arcefn.utils.plotting import plot_deta_dphi_plane, plot_representative_jet
from arcefn.utils.paths import EXPERIMENTS
TEST_H5 = EXPERIMENTS.parent / "data" / "top_tagging" / "test.h5"
MAX_K = 5
PHYS_PER_GROUP = 2000
PHYS_CAP = 15000
SUBCLASS_NAMES = {0: "QCD-Core", 1: "QCD-Edge", 2: "Top-Core", 3: "Top-Edge"}
# average panels: 2 class + 4 subclass (all)
PANELS = [
    (None, 0, "class_qcd", "QCD (all)"),
    (None, 1, "class_top", "Top (all)"),
    (0, None, "subclass_qcd_core", "QCD-Core"),
    (1, None, "subclass_qcd_edge", "QCD-Edge"),
    (2, None, "subclass_top_core", "Top-Core"),
    (3, None, "subclass_top_edge", "Top-Edge"),
]
def normalize(x): return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-10)
def load_centers(p):
    import torch
    try:
        import torch_directml  # noqa: F401
    except ImportError:
        pass
    sd = torch.load(p, map_location="cpu", weights_only=False)
    return normalize(sd["arcface_head.class_centers"].detach().cpu().numpy())
def stratified_sample(idx, n, rng):
    return idx if len(idx)<=n else rng.choice(idx, size=n, replace=False)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="experiments/robustness_s16_m05")
    parser.add_argument("--partition", type=str, default="predicted", choices=["predicted","true"],
                        help="Cluster per predicted (model-seen, primary) or true labels")
    args = parser.parse_args()
    model_dir = args.model_dir
    EMBEDDINGS = os.path.join(model_dir, "embeddings.npz")
    CHECKPOINT = os.path.join(model_dir, "checkpoint.pt")
    OUT_DIR = os.path.join(model_dir, "deta_dphi")
    t0=time.time()
    emb_np=np.load(EMBEDDINGS)
    embeddings, labels = emb_np["embeddings"], emb_np["labels"]
    assert len(embeddings)==404000
    centers=load_centers(CHECKPOINT)
    emb_norm=normalize(embeddings)
    preds = np.argmax(emb_norm @ centers.T, axis=1)
    partition_labels = preds if args.partition == "predicted" else labels
    # per-class clustering to get 4-way (predicted primary: what model sees)
    true_global=np.full(len(labels), -1, dtype=int)
    centroids={}
    for cls in (0,1):
        mask=partition_labels==cls
        sub,_=spectral_clustering_subclass(emb_norm[mask], max_k=MAX_K)
        cnt=Counter(sub)
        order=sorted(cnt, key=lambda c: -cnt[c])
        mapped=np.array([order.index(s) for s in sub])
        true_global[mask]=2*cls+mapped
        for sid in range(2):
            g=2*cls+sid
            sel=emb_norm[mask][mapped==sid]
            centroids[g]=normalize(sel.mean(axis=0, keepdims=True))[0]
    # also class centroids
    for cls in (0,1):
        centroids[f"cls{cls}"]=normalize(emb_norm[partition_labels==cls].mean(axis=0, keepdims=True))[0]
    # stratified sample for average planes
    rng=np.random.RandomState(42)
    parts=[]
    for sc in range(4):
        parts.append(stratified_sample(np.where(true_global==sc)[0], PHYS_PER_GROUP, rng))
    for cls in (0,1):
        parts.append(stratified_sample(np.where(partition_labels==cls)[0], PHYS_PER_GROUP, rng))
    phys_idx=np.unique(np.concatenate(parts))
    rng.shuffle(phys_idx)
    phys_idx=phys_idx[:PHYS_CAP]
    print(f"Loading {len(phys_idx)} physics events...")
    phys_events=load_events_by_indices(TEST_H5, phys_idx)
    os.makedirs(OUT_DIR, exist_ok=True)
    is_canonical = os.path.normpath(model_dir) == os.path.normpath("experiments/robustness_s16_m05")
    repo_root=os.path.dirname(os.path.dirname(model_dir))
    results={"n_total":404000, "n_physics":int(len(phys_idx)), "panels":{}}
    # average hist2d
    for subclass, cls, stem, label in PANELS:
        if subclass is not None:
            cond=true_global==subclass
        else:
            cond=partition_labels==cls
        n_total=int(cond.sum())
        panel_mask=cond[phys_idx]
        n_plot=int(panel_mask.sum())
        if n_plot==0: continue
        panel_events=phys_events[panel_mask]
        avg_path=os.path.join(OUT_DIR, f"{stem}_avg.png")
        plot_deta_dphi_plane(panel_events, label=label, save_path=avg_path)
        # B-minimal: paper keeps canonical 64-d; only canonical run copies to paper/figures
        if is_canonical:
            paper_path=os.path.join(repo_root,"paper","figures",f"deta_dphi_{stem}_avg.png")
            copy2(avg_path, paper_path)
        # representative: closest to centroid
        if subclass is not None:
            centroid=centroids[subclass]
            mask=true_global==subclass
            idx=np.where(mask)[0]
            # cosine to centroid
            sims=emb_norm[idx] @ centroid
            rep_idx=idx[np.argmax(sims)]
        else:
            centroid=centroids[f"cls{cls}"]
            idx=np.where(partition_labels==cls)[0]
            rep_idx=idx[np.argmax(emb_norm[idx] @ centroid)]
        # load single event
        rep_event=load_events_by_indices(TEST_H5, np.array([rep_idx]))
        rep_path=os.path.join(OUT_DIR, f"{stem}_rep.png")
        plot_representative_jet(rep_event[0], title=f"{label} representative (medoid)", save_path=rep_path)
        if is_canonical:
            copy2(rep_path, os.path.join(repo_root,"paper","figures",f"deta_dphi_{stem}_rep.png"))
        results["panels"][stem]={"label":label,"n_total":n_total,"n_plotted":n_plot,"rep_idx":int(rep_idx)}
        print(f"  {stem}: n_total={n_total} n_plot={n_plot} rep={rep_idx}")
    # per-model json + canonical copy for REPRODUCING
    out_json=os.path.join(OUT_DIR, "deta_dphi_planes.json")
    with open(out_json,"w") as f: json.dump(results,f,indent=2)
    if is_canonical:
        canon_json=os.path.join(os.path.dirname(model_dir), "deta_dphi_planes.json")
        with open(canon_json,"w") as f: json.dump(results,f,indent=2)
        print(f"Saved {canon_json} + {out_json} ({time.time()-t0:.0f}s) [COMPLETE]")
    else:
        print(f"Saved {out_json} ({time.time()-t0:.0f}s) [COMPLETE]")
if __name__=="__main__": main()
