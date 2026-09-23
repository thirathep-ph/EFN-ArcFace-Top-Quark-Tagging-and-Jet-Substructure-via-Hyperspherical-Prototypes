import os
import numpy as np
import matplotlib.pyplot as plt

def main():
    embeddings_path = 'experiments/arcface_100ep/embeddings.npz'
    output_plot_path = 'experiments/svd_analysis/svd_scree_plot.png'
    
    print("1. Loading embeddings...")
    data = np.load(embeddings_path)
    embeddings = data['embeddings']
    
    # Normalize embeddings (ArcFace spherical projection)
    X = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
    print(f"Loaded {X.shape[0]:,} embeddings of dimension {X.shape[1]}")
    
    # 2. SVD of the embedding matrix
    print("2. Computing Singular Value Decomposition (SVD) on CPU...")
    # X shape is (N, d). SVD of X: U, S, Vt
    # We only need singular values S
    _, S, _ = np.linalg.svd(X, full_matrices=False)
    
    # 3. Calculate Variance Explained
    # The variance explained by each singular value is S_i^2 / sum(S_j^2)
    squared_S = S ** 2
    total_var = np.sum(squared_S)
    variance_explained = squared_S / total_var
    cumulative_variance = np.cumsum(variance_explained)
    
    # 4. Find threshold indices
    thresholds = [0.90, 0.95, 0.99, 0.999]
    dims_at_thresholds = {}
    for t in thresholds:
        idx = np.where(cumulative_variance >= t)[0][0] + 1
        dims_at_thresholds[t] = idx
        
    print("\n" + "="*50)
    print("RESULTS: SVD EXPLAINED VARIANCE ANALYSIS")
    print("="*50)
    print(f"Nominal Dimensions                  : 64")
    print(f"Dimensions explaining 90.0% variance: {dims_at_thresholds[0.90]:d}")
    print(f"Dimensions explaining 95.0% variance: {dims_at_thresholds[0.95]:d}")
    print(f"Dimensions explaining 99.0% variance: {dims_at_thresholds[0.99]:d}")
    print(f"Dimensions explaining 99.9% variance: {dims_at_thresholds[0.999]:d}")
    print("="*50)
    
    print("\nSingular Value Profile (Top 10):")
    for i in range(10):
        print(f"  Dim {i+1:2d}: Singular Value = {S[i]:.2f} | Var = {variance_explained[i]*100:.2f}% | Cum = {cumulative_variance[i]*100:.2f}%")
        
    # 5. Plot Scree Plot & Cumulative Variance
    print(f"\n3. Saving SVD Scree Plot to '{output_plot_path}'...")
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    color = '#1f77b4'
    ax1.set_xlabel('Singular Value Index')
    ax1.set_ylabel('Singular Value magnitude', color=color)
    ax1.plot(range(1, len(S)+1), S, 'o-', color=color, linewidth=2, label='Singular Values')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.15)
    
    ax2 = ax1.twinx()  
    color = '#ff7f0e'
    ax2.set_ylabel('Cumulative Explained Variance', color=color)
    ax2.plot(range(1, len(cumulative_variance)+1), cumulative_variance, 's--', color=color, linewidth=1.5, alpha=0.8, label='Cumulative Variance')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Draw thresholds
    for t, dim in dims_at_thresholds.items():
        ax2.axvline(x=dim, color='gray', linestyle=':', alpha=0.5)
        ax2.annotate(f"{t*100:.0f}% @ {dim}d", xy=(dim, t), xytext=(dim+1.5, t-0.03),
                     arrowprops=dict(arrowstyle="->", color='gray', alpha=0.5),
                     fontsize=9, color='darkred')
                     
    plt.title('SVD Scree Plot & Cumulative Explained Variance', fontsize=13)
    fig.tight_layout()  
    plt.savefig(output_plot_path, dpi=150)
    plt.close()
    print("Scree Plot saved successfully.")

if __name__ == '__main__':
    main()
