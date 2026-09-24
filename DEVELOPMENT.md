# DEVELOPMENT.md - Project Structure & Architecture Guide

## Model Architecture

The project uses **EFN (Energy Flow Network) + ArcFace** for ante-hoc interpretable jet tagging.

```
Constituents (E, px, py, pz per particle)
    |
    +---> Centering/Rotation: DeltaR_i = (Delta_eta_i, Delta_phi_i)
    |
    +---> EFN (Energy Flow Network) - IRC-Safe
    |     +-- Phi MLP:  [2 -> particle_dim -> particle_dim -> particle_dim]
    |     +-- Sum Pooling weighted by energy fraction z_i = E_i / Sum E
    |     +-- Rho MLP:   [particle_dim -> particle_dim -> embedding_dim]
    |
    +---> Jet Latent Embedding  R^embedding_dim  (L2-normalized to hypersphere)
    |
    +---> ArcFace Classification Head
          +-- class_centers: nn.Parameter (n_classes, embedding_dim)
          +-- Cosine similarity -> geodesic margin m, scale s
```

### Key Model Attributes

- `model.arcface_head.class_centers` — Class centers on the hypersphere (NOT `prototype_layer.prototypes`)
- `model.efn` — EFN backbone
- Output: `(logits, similarities, jet_embedding)`

## Clustering

Subclass discovery uses **Spectral Clustering + Eigengap Heuristic** (NOT DPGMM).

Single source of truth: `arcefn/utils/clustering.py`

```python
from arcefn.utils.clustering import estimate_clusters_eigengap, spectral_clustering_subclass

# Returns (labels, n_clusters)
labels, k = spectral_clustering_subclass(embeddings, max_k=10)
```

## Project Structure

```
arcefn/                        # Core library (editable package)
+-- models/
|   +-- efn_arcface.py              # EFN + ArcFace model (main)
|   +-- efn_linear.py               # EFN + Linear baseline
|   +-- efn_coslinear.py            # EFN + Cosine Linear baseline
+-- data/
|   +-- loader.py                   # H5 -> Awkward -> PyTorch data pipeline
+-- utils/
    +-- clustering.py               # Unified Spectral Clustering + Eigengap
    +-- physics.py                  # FastJet substructure: mass, tau32, zg, Lund
    +-- plotting.py                 # Visualization: latent space, Lund planes, ROC
    +-- device.py                   # Device detection, AMP support

scripts/                            # Training & evaluation
+-- train_arcface.py                 # EFN + ArcFace training (main)
+-- train_baselines.py               # Baseline model training (Linear + CosLinear)
+-- evaluate_model.py                # Standalone model evaluation
+-- evaluate_robustness.py           # Robustness sweep evaluation
+-- run_pipeline.py                  # Orchestration script (runs all analysis)
+-- download_data.py                 # Dataset download
+-- analysis/                        # Analysis scripts (compute_/generate_/fit_/poc_)
|   +-- compute_*.py                 # Metric computations (30+ scripts)
|   +-- generate_*.py                # Figure generators
|   +-- fit_*.py                     # PySR symbolic regression
|   +-- poc_*.py                     # Proof-of-concept / exploratory
|   +-- regenerate_stale_figures.py  # Re-generates result figures
+-- scratch/                         # One-off scripts (fixes, combiners)

experiments/                        # Trained model outputs
+-- robustness_s*_m*/               # Robustness trials (scale, margin) — s16_m05 = main
+-- arcface_seed_*/                 # Multi-seed ArcFace runs (42, 123, 7)
+-- baselines_linear_seed_*/        # Linear baseline seeds
+-- baselines_coslinear_seed_*/     # CosLinear baseline seeds
+-- physics_sr/                     # Symbolic regression artifacts (JSON + npz)
+-- archive/                        # Superseded artifacts (do not cite)
+-- logs/                           # Training and PySR logs
+-- svd_analysis/                   # SVD dimension analysis
+-- eda/                            # EDA statistics
+-- esr/                            # Full-embedding symbolic regression POC
```

## Training Modes

- **`--no-chunk`**: Loads entire dataset into RAM (~2.4GB). Faster, recommended.
- **Default (chunked)**: Reads H5 in chunks. Memory-safe, slower.
- **`--amp`**: Automatic Mixed Precision (FP16). ~2x speedup on compatible GPUs. **CUDA only — do not use on this machine (AMD RX 5700 XT via torch-directml has no AMP support).**
- **`--train-only`**: Skip heavy analysis (UMAP, clustering, Lund planes).
- **`--analysis_only`**: Skip training, run analysis on existing checkpoint.

## Common Patterns

### Loading a trained model

```python
from arcefn.models.efn_arcface import TopTaggingModel

model = TopTaggingModel(s=16.0, m=0.0, embedding_dim=64, particle_dim=128)
state_dict = torch.load('checkpoints/best_model_efn.pt', map_location='cpu')
model.load_state_dict(state_dict)
model.eval()
```

### Extracting class centers

```python
class_centers = model.arcface_head.class_centers.detach().cpu().numpy()
```

### Running clustering on embeddings

```python
from arcefn.utils.clustering import spectral_clustering_subclass

embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
labels, k = spectral_clustering_subclass(embeddings_norm, max_k=10)
```

## Important Notes

- The model uses `arcface_head.class_centers` (not `prototype_layer.prototypes`)
- Clustering uses Spectral Clustering + Eigengap (not DPGMM)
- The project uses `--no-chunk` mode for training (loads all data into RAM)
- Main model: `experiments/robustness_s16_m05/checkpoint.pt` (seed 42, 50 epochs, val_acc ~0.9179)
- Multi-seed runs exist: seeds 42, 123, 7 — subclass structure is seed-stable (global ARI ~0.87)
- See docs/RESULTS.md for the full results record
