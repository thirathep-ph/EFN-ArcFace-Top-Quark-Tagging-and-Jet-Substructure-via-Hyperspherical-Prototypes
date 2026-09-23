"""
Plotting Utilities for Jet Tagging Analysis
===========================================
- Latent Space Visualization (t-SNE)
- ROC & Background Rejection
- Representative Jet Display
- Lund Plane Visualization
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.manifold import TSNE
import awkward as ak
import fastjet


def plot_latent_space(embeddings, labels, class_centers=None, save_path='results/latent_space.png'):
    """Plot t-SNE of the latent space including class centers."""
    print("Computing t-SNE for latent space and class centers...")
    
    n_emb = len(embeddings)
    
    # L2-normalize embeddings and class centers to unit hypersphere
    # since ArcFace operates in angular/cosine similarity space
    eps = 1e-10
    embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + eps)
    
    if class_centers is not None:
        centers_norm = class_centers / (np.linalg.norm(class_centers, axis=1, keepdims=True) + eps)
        combined = np.concatenate([embeddings_norm, centers_norm], axis=0)
    else:
        combined = embeddings_norm

    # Filter out any NaNs
    mask = ~np.any(np.isnan(combined), axis=1)
    combined = combined[mask]
    
    tsne = TSNE(n_components=2, random_state=42)
    combined_2d = tsne.fit_transform(combined)
    
    latent_2d = combined_2d[:n_emb]
    centers_2d = combined_2d[n_emb:] if class_centers is not None else None
    
    plt.figure(figsize=(11, 9))
    # Plot jet embeddings
    sns.scatterplot(x=latent_2d[:, 0], y=latent_2d[:, 1], hue=labels, 
                    palette='viridis', alpha=0.4, s=15)
    
    # Plot class centers as distinct markers
    if centers_2d is not None:
        n_c = len(centers_2d)
        plt.scatter(centers_2d[:n_c//2, 0], centers_2d[:n_c//2, 1], 
                    marker='*', s=300, color='red', edgecolors='black', label='QCD Center', zorder=5)
        plt.scatter(centers_2d[n_c//2:, 0], centers_2d[n_c//2:, 1], 
                    marker='*', s=300, color='cyan', edgecolors='black', label='Top Center', zorder=5)

    plt.title("EFN Latent Space & ArcFace Class Centers")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.2)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved updated latent space plot to {save_path}")

def plot_performance(labels, sims, save_path='results/performance.png'):
    """Plot ROC and Background Rejection."""
    # sims here is expected to be (N, 2) or (N,)
    if len(sims.shape) > 1:
        # Use Top class similarity for ROC (assuming index 1 is Top)
        scores = sims[:, 1]
    else:
        scores = sims

    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # 1. ROC Curve
    ax1.plot(fpr, tpr, color='crimson', lw=3, label=f'EFN+ArcFace (AUC = {roc_auc:.4f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_xlabel('False Positive Rate (QCD Mistag)')
    ax1.set_ylabel('True Positive Rate (Top Efficiency)')
    ax1.set_title('ROC Curve Performance')
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    
    # 2. Background Rejection (1/FPR)
    # Filter to avoid div by zero
    mask = fpr > 0
    tpr_m = tpr[mask]
    rej_m = 1.0 / fpr[mask]
    
    ax2.plot(tpr_m, rej_m, color='teal', lw=3, label='Background Rejection')
    ax2.set_yscale('log')
    ax2.set_xlabel('Signal Efficiency (TPR)')
    ax2.set_ylabel('Background Rejection (1/FPR)')
    ax2.set_title('Background Rejection at Fixed Efficiency')
    ax2.grid(True, which="both", ls="-", alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved performance plots to {save_path}")

def plot_representative_jet(event_p4, title="Representative Jet", save_path=None):
    """Plot jet constituents in eta-phi plane with energy weighting."""
    px = ak.to_numpy(event_p4.px)
    py = ak.to_numpy(event_p4.py)
    pz = ak.to_numpy(event_p4.pz)
    e = ak.to_numpy(event_p4.E)
    
    pt = np.sqrt(px**2 + py**2)
    eta = np.arcsinh(pz / (pt + 1e-10))
    phi = np.arctan2(py, px)
    
    # jet axis for Δη, Δφ
    jet_eta = np.arcsinh(np.sum(pz) / (np.sqrt(np.sum(px)**2 + np.sum(py)**2) + 1e-10))
    jet_phi = np.arctan2(np.sum(py), np.sum(px))
    deta = eta - jet_eta
    dphi = (phi - jet_phi + np.pi) % (2 * np.pi) - np.pi
    
    plt.figure(figsize=(9, 7))
    # Size reflects pT, color reflects energy fraction
    z = e / (np.sum(e) + 1e-10)
    scatter = plt.scatter(deta, dphi, s=pt*8, alpha=0.7, c=z, cmap='plasma', edgecolors='black', linewidth=0.5)
    plt.colorbar(scatter, label='Energy Fraction (z)')
    plt.xlabel(r'$\Delta\eta$')
    plt.ylabel(r'$\Delta\phi$')
    plt.title(f"{title}\n(Constituent Pattern)", fontsize=14)
    plt.grid(True, alpha=0.2)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_deta_dphi_plane(events, label="Jets", save_path='results/deta_dphi.png'):
    """Δη-Δφ plane: 2D histogram of constituents relative to jet axis."""
    # vectorized Δη, Δφ per constituent
    px = ak.to_numpy(ak.flatten(events.px))
    py = ak.to_numpy(ak.flatten(events.py))
    pz = ak.to_numpy(ak.flatten(events.pz))
    e = ak.to_numpy(ak.flatten(events.E))
    # jet-level axis per jet (broadcast via counts)
    counts = ak.to_numpy(ak.num(events))
    jet_px = np.repeat(ak.to_numpy(ak.sum(events.px, axis=1)), counts)
    jet_py = np.repeat(ak.to_numpy(ak.sum(events.py, axis=1)), counts)
    jet_pz = np.repeat(ak.to_numpy(ak.sum(events.pz, axis=1)), counts)
    jet_pt = np.sqrt(jet_px**2 + jet_py**2)
    jet_eta = np.arcsinh(jet_pz / (jet_pt + 1e-10))
    jet_phi = np.arctan2(jet_py, jet_px)
    pt = np.sqrt(px**2 + py**2)
    eta = np.arcsinh(pz / (pt + 1e-10))
    phi = np.arctan2(py, px)
    deta = eta - jet_eta
    dphi = (phi - jet_phi + np.pi) % (2 * np.pi) - np.pi
    # mask padded zeros
    mask = e > 1e-8
    deta, dphi, e = deta[mask], dphi[mask], e[mask]
    if len(deta) == 0:
        print(f"Warning: No constituents for {label}")
        return
    z = e / (np.sum(e) + 1e-10) if False else None  # unweighted for density
    plt.figure(figsize=(9, 7))
    plt.hist2d(deta, dphi, bins=50, cmap='viridis', cmin=1)
    plt.colorbar(label='Constituent density')
    plt.xlabel(r'$\Delta\eta$')
    plt.ylabel(r'$\Delta\phi$')
    plt.title(f"$\\Delta\\eta$-$\\Delta\\phi$ plane ({label})")
    plt.grid(True, alpha=0.1)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved deta-dphi plane to {save_path}")

def plot_lund_plane(events, label="Jets", save_path='results/lund_plane.png'):
    """
    Accurate Lund Plane using Iterative C/A de-clustering.
    x = ln(1/Delta), y = ln(kt)
    """
    from arcefn.utils.physics import get_lund_coordinates
    
    print(f"Computing Lund Plane for {label}...")
    ln_inv_delta, ln_kt = get_lund_coordinates(events)
    
    if len(ln_inv_delta) == 0:
        print(f"Warning: No Lund coordinates found for {label}")
        return

    plt.figure(figsize=(9, 7))
    plt.hist2d(ln_inv_delta, ln_kt, bins=50, cmap='viridis', cmin=1)
    plt.colorbar(label='Splitting Density')
    plt.xlabel(r'$\ln(1/\Delta)$')
    plt.ylabel(r'$\ln(k_t)$')
    plt.title(f"Lund Plane ({label})")
    plt.grid(True, alpha=0.1)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved physical Lund Plane plot to {save_path}")

def plot_training_results(history, cm, save_path='results/training_results_efn.png'):
    """Plot Loss, Accuracy, Prototype Separation, and Confusion Matrix."""
    plt.figure(figsize=(16, 12))
    
    # 1. Loss
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='blue', lw=2)
    plt.plot(history['val_loss'], label='Val Loss', color='orange', lw=2)
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.1)
    
    # 2. Accuracy
    plt.subplot(2, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc', color='green', lw=2)
    plt.plot(history['val_acc'], label='Val Acc', color='red', lw=2)
    plt.title("Training & Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.1)
    
    # 3. Class Center Separation
    plt.subplot(2, 2, 3)
    plt.plot(history['sep_qcd'], label='QCD Sim', color='purple', alpha=0.7)
    plt.plot(history['sep_top'], label='Top Sim', color='darkorange', alpha=0.7)
    plt.title("Latent Space Compression (Avg Cosine Similarity)")
    plt.xlabel("Epoch")
    plt.ylabel("Similarity to Class Center")
    plt.legend()
    plt.grid(True, alpha=0.1)
    
    # 4. Confusion Matrix
    plt.subplot(2, 2, 4)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    
    # Save standalone CM for better visibility
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix (Detailed)")
    out_dir = os.path.dirname(save_path) if os.path.dirname(save_path) else '.'
    plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close('all')

def plot_decision_boundary(labels, class_scores, save_path='results/decision_boundary.png'):
    """Visualize the ArcFace decision boundary on the Similarity Plane."""
    plt.figure(figsize=(8, 8))
    qcd_sim = class_scores[:, 0]
    top_sim = class_scores[:, 1]
    
    # Plot points
    plt.scatter(qcd_sim[labels==0], top_sim[labels==0], s=10, alpha=0.3, label='True QCD', color='blue')
    plt.scatter(qcd_sim[labels==1], top_sim[labels==1], s=10, alpha=0.3, label='True Top', color='red')
    
    # Plot the diagonal (Decision Boundary for ArcFace)
    lims = [min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1])]
    plt.plot(lims, lims, 'k--', lw=2, label='Decision Boundary (Sim_Q = Sim_T)')
    
    plt.xlabel("Max Similarity to QCD Center")
    plt.ylabel("Max Similarity to Top Center")
    plt.title("ArcFace Decision Boundary & Cluster Separation")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_feature_distributions(labels, features, feature_names, save_path='results/feature_distributions.png'):
    """
    Plot statistical distributions of physics features for Top vs QCD.
    """
    n_features = len(feature_names)
    rows = (n_features + 2) // 3
    cols = min(n_features, 3)
    
    plt.figure(figsize=(cols * 6, rows * 5))
    qcd_mask = labels == 0
    top_mask = labels == 1
    
    for i, name in enumerate(feature_names):
        plt.subplot(rows, cols, i + 1)
        sns.kdeplot(features[qcd_mask, i], label='QCD', fill=True, color='blue', alpha=0.4)
        sns.kdeplot(features[top_mask, i], label='Top', fill=True, color='red', alpha=0.4)
        plt.title(f"Distribution: {name}", fontsize=14)
        plt.xlabel(name)
        plt.legend()
        plt.grid(True, alpha=0.1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved feature distribution analysis to {save_path}")

def plot_physics_correlations(labels, scores, features, feature_names, save_path='results/correlations.png'):
    """
    Plot correlation between model scores and physical observables.
    Helps detect mass sculpting or reliance on specific features.
    """
    import pandas as pd
    
    data = pd.DataFrame(features, columns=feature_names)
    data['Model Score'] = scores
    data['Class'] = ['Top' if l == 1 else 'QCD' for l in labels]
    
    # Calculate correlations
    corr = data.drop(columns=['Class']).corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Correlation: Model Decision vs. Physical Observables", fontsize=15)
    
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    # Also plot Model Score vs Mass specifically (Mass Sculpting check)
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=data, x='Mass', y='Model Score', hue='Class', fill=True, alpha=0.5)
    plt.title("Model Decision vs. Jet Mass (Mass Sculpting Check)")
    out_dir = os.path.dirname(save_path) if os.path.dirname(save_path) else '.'
    plt.savefig(os.path.join(out_dir, 'mass_sculpting.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved correlation analysis to {save_path} and {os.path.join(out_dir, 'mass_sculpting.png')}")
