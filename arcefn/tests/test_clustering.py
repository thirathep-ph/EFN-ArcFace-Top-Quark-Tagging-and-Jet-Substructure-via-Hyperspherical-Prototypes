import numpy as np


def test_estimate_clusters_eigengap():
    from arcefn.utils.clustering import estimate_clusters_eigengap
    np.random.seed(42)
    centers = [np.array([1, 0, 0]), np.array([-0.5, 0.866, 0]), np.array([-0.5, -0.866, 0])]
    X = np.concatenate([np.random.randn(100, 3) * 0.1 + c for c in centers])
    optimal_k, gap = estimate_clusters_eigengap(X, max_k=10)
    assert optimal_k == 3
    assert np.all(gap[1:] > 0)  # gap[0] for k=1 is always 0


def test_spectral_clustering_subclass():
    from arcefn.utils.clustering import spectral_clustering_subclass
    np.random.seed(42)
    centers = [np.array([0.5] * 64), np.array([-0.5] * 64)]
    X = np.concatenate([np.random.randn(200, 64) * 0.3 + c for c in centers])
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
    labels, k = spectral_clustering_subclass(X_norm, max_k=10)
    assert k >= 2
    assert len(labels) == 400
    assert len(np.unique(labels)) == k


def test_eigengap_single_cluster():
    from arcefn.utils.clustering import estimate_clusters_eigengap
    np.random.seed(42)
    X = np.random.randn(100, 64) * 0.01
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
    optimal_k, gap = estimate_clusters_eigengap(X_norm, max_k=10)
    assert optimal_k >= 1
    assert optimal_k <= 10
