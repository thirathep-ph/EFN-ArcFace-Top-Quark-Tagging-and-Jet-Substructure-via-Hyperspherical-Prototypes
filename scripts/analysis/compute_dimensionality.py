"""
Dimensionality Metrics: TwoNN, Participation Ratio, Class Center Angle
=======================================================================
Computes dimensionality metrics from Table 4 of the paper:

1. **TwoNN intrinsic dimension** ??" nearest-neighbor ratio estimator.
   Expected: ~2.17
2. **Participation Ratio** ??" from PCA eigenvalues.
   Expected: ~1.38
3. **Class Center Angular Distance** ??" arccos of cosine similarity between
   ``W_QCD`` and ``W_Top`` from the trained model.
   Expected: ~0.76 rad

Output
------
* ``EXPERIMENTS / dimensionality_results.json``
* Results also printed to stdout in a formatted table.
"""
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

from arcefn.utils.paths import DATA_DIR, EXPERIMENTS, CHECKPOINT
from arcefn.utils.device import get_device
from arcefn.data.loader import load_awkward, JetTaggingDataset
from arcefn.models.efn_arcface import TopTaggingModel

N_EVENTS = 50000
BATCH_SIZE = 2048


def load_model_and_embeddings(device, max_events=N_EVENTS):
    """
    Load the trained model and compute normalised embeddings on the test set.

    Parameters
    ----------
    device : torch.device
        Compute device.
    max_events : int
        Maximum number of events to use.

    Returns
    -------
    embeddings : ndarray, shape (N, D)
        L2-normalised embeddings.
    labels : ndarray, shape (N,)
        Ground-truth labels.
    """
    model = TopTaggingModel(s=16.0, m=0.5, embedding_dim=64, particle_dim=128).to(device)
    sd = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(sd, strict=True)
    model.eval()

    test_h5 = DATA_DIR / 'test.h5'
    print(f"Loading {max_events:,} test events ...")
    events, labels, weights = load_awkward(str(test_h5), max_events=max_events, lazy=False)

    ds = JetTaggingDataset(events, labels, weights)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_emb = []
    all_lab = []
    with torch.no_grad():
        for x, y, w, m, _ in loader:
            x, m = x.to(device), m.to(device)
            _, _, jet_emb = model(x, mask=m)
            emb = F.normalize(jet_emb, p=2, dim=1)
            all_emb.append(emb.cpu().numpy())
            all_lab.append(y.numpy())

    embeddings = np.concatenate(all_emb)
    labels = np.concatenate(all_lab)
    print(f"  Embeddings shape: {embeddings.shape}")
    return embeddings, labels


# ── TwoNN ────────────────────────────────────────────────────────────

def compute_twonn_dimension(embeddings, n_neighbors=3, random_state=42):
    """
    Estimate intrinsic dimension using the TwoNN estimator (Facco et al.).

    For each point, let :math:`r_{i,1}` and :math:`r_{i,2}` be the distances
    to its first and second nearest neighbours.  The estimator is

    .. math::
        d = \\frac{n}{\\sum_i \\log\\left(r_{i,2} / r_{i,1}\\right)}

    Parameters
    ----------
    embeddings : ndarray, shape (N, D)
        L2-normalised embeddings.
    n_neighbors : int
        Number of neighbours to compute (self + 1st + 2nd NN; must be ≥ 3).
    random_state : int
        Seed for the NN algorithm.

    Returns
    -------
    dimension : float
        Estimated intrinsic dimension.
    """
    n = len(embeddings)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine').fit(embeddings)
    distances, _ = nbrs.kneighbors(embeddings)

    # distances[:, 0] = 0 (self), distances[:, 1] = 1st NN, distances[:, 2] = 2nd NN
    r1 = distances[:, 1] + 1e-10
    r2 = distances[:, 2] + 1e-10
    mu = r2 / r1

    d = n / np.sum(np.log(mu))
    return float(d)


# ── Participation Ratio ──────────────────────────────────────────────

def compute_participation_ratio(embeddings):
    """
    Compute the participation ratio from PCA eigenvalues.

    .. math::
        PR = \\frac{(\\sum_i \\lambda_i)^2}{\\sum_i \\lambda_i^2}

    where :math:`\\lambda_i` are the eigenvalues of the covariance matrix.

    Parameters
    ----------
    embeddings : ndarray, shape (N, D)
        L2-normalised embeddings.

    Returns
    -------
    pr : float
        Participation ratio.
    """
    pca = PCA().fit(embeddings)
    evals = pca.explained_variance_
    pr = (np.sum(evals) ** 2) / np.sum(evals ** 2)
    return float(pr)


# ── Class Center Angular Distance ────────────────────────────────────

def compute_class_center_angle(device):
    """
    Load the trained model checkpoint and compute the angular distance
    between ``W_QCD`` and ``W_Top`` class centres.

    Returns
    -------
    angle_rad : float
        Angular distance in radians.
    angle_deg : float
        Angular distance in degrees.
    cosine_sim : float
        Cosine similarity between centres.
    """
    model = TopTaggingModel(s=16.0, m=0.5, embedding_dim=64, particle_dim=128).to(device)
    sd = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(sd, strict=True)

    centers = model.arcface_head.class_centers.detach().cpu().numpy()
    w_qcd = centers[0]
    w_top = centers[1]

    # Normalise
    w_qcd = w_qcd / (np.linalg.norm(w_qcd) + 1e-10)
    w_top = w_top / (np.linalg.norm(w_top) + 1e-10)

    cos_sim = float(np.clip(np.dot(w_qcd, w_top), -1.0, 1.0))
    angle_rad = float(np.arccos(cos_sim))
    angle_deg = float(np.degrees(angle_rad))

    return angle_rad, angle_deg, cos_sim


# ── Main ────────────────────────────────────────────────────────────

def main():
    """
    Run all dimensionality metrics and print a formatted results table.

    Results are saved to ``EXPERIMENTS / dimensionality_results.json``.
    """
    print("=" * 60)
    print("Dimensionality Metrics (Table 3)")
    print("=" * 60)

    device, _ = get_device()

    # 1. TwoNN
    print("\n--- TwoNN Intrinsic Dimension ---")
    t0 = time.time()
    embeddings, labels = load_model_and_embeddings(device)
    d_twonn = compute_twonn_dimension(embeddings)
    print(f"  TwoNN dimension: {d_twonn:.4f}  ({time.time()-t0:.1f}s)")

    # 2. Participation Ratio
    print("\n--- Participation Ratio ---")
    t0 = time.time()
    pr = compute_participation_ratio(embeddings)
    print(f"  Participation Ratio: {pr:.4f}  ({time.time()-t0:.1f}s)")

    # 3. Class Center Angular Distance
    print("\n--- Class Center Angular Distance ---")
    t0 = time.time()
    angle_rad, angle_deg, cos_sim = compute_class_center_angle(device)
    print(f"  Cosine similarity: {cos_sim:.4f}")
    print(f"  Angular distance:  {angle_rad:.4f} rad = {angle_deg:.2f}°  ({time.time()-t0:.1f}s)")

    # ── Results ─────────────────────────────────────────────────
    results = {
        'twonn_intrinsic_dimension': d_twonn,
        'participation_ratio': pr,
        'class_center_angular_distance_rad': angle_rad,
        'class_center_angular_distance_deg': angle_deg,
        'class_center_cosine_similarity': cos_sim,
        'n_events': len(embeddings),
        'embedding_dim': embeddings.shape[1],
    }

    # ── Print table ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  {'Metric':<40s} {'Value':>10s}")
    print("  " + "-" * 52)
    print(f"  {'TwoNN Intrinsic Dimension':<40s} {d_twonn:>10.4f}")
    print(f"  {'Participation Ratio':<40s} {pr:>10.4f}")
    print(f"  {'Class Center Angular Distance (rad)':<40s} {angle_rad:>10.4f}")
    print(f"  {'Class Center Angular Distance (deg)':<40s} {angle_deg:>10.2f}")
    print(f"  {'Class Center Cosine Similarity':<40s} {cos_sim:>10.4f}")
    print(f"  {'Number of events':<40s} {len(embeddings):>10,}")
    print(f"  {'Embedding dimension':<40s} {embeddings.shape[1]:>10d}")
    print("=" * 60)

    # ── Save JSON ────────────────────────────────────────────────
    out_path = EXPERIMENTS / 'dimensionality_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print("\n[COMPLETE]")


if __name__ == '__main__':
    main()
