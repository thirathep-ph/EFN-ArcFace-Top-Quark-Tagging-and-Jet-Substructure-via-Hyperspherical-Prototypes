"""
Unified Clustering Module for Jet Tagging
=========================================
Single source of truth for Spectral Clustering + Eigengap heuristic.

All scripts should import from here to avoid code duplication and ensure
consistent clustering behavior across the project.

Functions:
    estimate_clusters_eigengap(X_norm, max_k=10) -> (optimal_k, eigenvalues)
    spectral_clustering_subclass(X, max_k=10) -> (labels, n_clusters)
"""
import numpy as np
from scipy.sparse.linalg import eigsh, LinearOperator
from sklearn.cluster import KMeans


def estimate_clusters_eigengap(X_norm: np.ndarray, max_k: int = 10) -> tuple[int, np.ndarray | None]:
    """
    Estimates the optimal number of clusters using the eigengap heuristic.
    Uses parameter-free global Cosine Similarity and sparse ARPACK solver.

    Parameters
    ----------
    X_norm : np.ndarray, shape (n_samples, n_features)
        L2-normalized feature matrix (cosine similarity = dot product).
    max_k : int
        Maximum number of clusters to check.

    Returns
    -------
    optimal_k : int
        Estimated optimal number of clusters.
    eigenvalues : np.ndarray or None
        Sorted eigenvalues of the normalized Laplacian, or None on failure.
    """
    n_samples = X_norm.shape[0]
    if n_samples < 3:
        return 2, None

    # Global Cosine Similarity Matrix: A_ij = (1 + cos(theta_ij)) / 2
    # Row sums: d_i = (N + X_norm_i . sum_j X_norm_j) / 2
    sum_X = np.sum(X_norm, axis=0)
    d = (n_samples + X_norm @ sum_X) / 2.0
    d = np.clip(d, 1e-10, None)
    d_inv_sqrt = 1.0 / np.sqrt(d)

    # LinearOperator for normalized Laplacian M = D^{-1/2} A D^{-1/2}
    def matvec(v):
        y = d_inv_sqrt * v
        Ay = (np.sum(y) + X_norm @ (X_norm.T @ y)) / 2.0
        return d_inv_sqrt * Ay

    M_op = LinearOperator((n_samples, n_samples), matvec=matvec)

    try:
        k_eigen = min(max_k + 2, n_samples - 2)
        # M = D^{-1/2} A D^{-1/2} is PSD: A_ij = (1 + cos(theta_ij)) / 2 is a
        # Gram-based affinity, so all eigenvalues are non-negative and
        # `which='LM'` returns the largest eigenvalues of M (equivalently the
        # smallest eigenvalues of L_norm) with no negative-value interleaving.
        vals, _ = eigsh(M_op, k=k_eigen, which='LM')

        # eigenvalues of L_norm = 1.0 - vals, sorted in ascending order
        eigenvalues = np.sort(1.0 - vals)

        max_k_checked = min(max_k, len(eigenvalues) - 2)
        if max_k_checked < 2:
            return 2, eigenvalues
        # Gaps between consecutive non-trivial eigenvalues lambda_1..lambda_{max_k};
        # argmax + 2 maps the first gap (lambda_2 - lambda_1) to k=2, capped at max_k.
        gaps = np.diff(eigenvalues[1:max_k_checked + 1])
        optimal_k = int(np.argmax(gaps) + 2)
        return optimal_k, eigenvalues
    except Exception as e:
        print(f"Warning in eigengap calculation: {e}. Falling back to default k=5.")
        return 5, None


def spectral_clustering_subclass(X: np.ndarray, max_k: int = 10) -> tuple[np.ndarray, int]:
    """
    Standard Spectral Clustering using parameter-free global Cosine Similarity.
    Uses eigengap heuristic to determine optimal k, then K-means on eigenvectors.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix (will be L2-normalized internally).
    max_k : int
        Maximum number of clusters to check.

    Returns
    -------
    labels : np.ndarray, shape (n_samples,)
        Cluster assignments.
    n_clusters : int
        Number of clusters found.
    """
    n_samples = X.shape[0]
    if n_samples < 2:
        return np.zeros(n_samples, dtype=int), 1

    # L2-normalize to ensure Cosine distance is used
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)

    # Step 1: Eigengap Heuristic to find optimal k
    optimal_k, _ = estimate_clusters_eigengap(X_norm, max_k=max_k)
    print(f"  Eigengap heuristic suggests optimal subclasses k = {optimal_k}")

    # Step 2: Spectral Clustering with global cosine similarity
    sum_X = np.sum(X_norm, axis=0)
    d = (n_samples + X_norm @ sum_X) / 2.0
    d = np.clip(d, 1e-10, None)
    d_inv_sqrt = 1.0 / np.sqrt(d)

    def matvec(v):
        y = d_inv_sqrt * v
        Ay = (np.sum(y) + X_norm @ (X_norm.T @ y)) / 2.0
        return d_inv_sqrt * Ay

    M_op = LinearOperator((n_samples, n_samples), matvec=matvec)

    try:
        vals, vecs = eigsh(M_op, k=optimal_k, which='LM')

        # Sort descending by eigenvalues of M
        idx = np.argsort(vals)[::-1]
        vecs = vecs[:, idx]

        # Normalize rows of eigenvectors
        row_norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        U = vecs / (row_norms + 1e-10)

        # Cluster using K-means
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(U)

        return labels, optimal_k
    except Exception as e:
        print(f"Warning in spectral clustering: {e}. Falling back to standard K-means.")
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_norm)
        return labels, optimal_k
