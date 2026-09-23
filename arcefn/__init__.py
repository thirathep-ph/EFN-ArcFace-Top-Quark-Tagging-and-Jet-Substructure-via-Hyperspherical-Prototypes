"""
arcefn: Ante-hoc Interpretable Jet Tagging with EFN and ArcFace
================================================================
EFN (Energy Flow Network) backbone + ArcFace hyperspherical prototypes
for intrinsically interpretable top quark tagging.

Core components:
    - TopTaggingModel: Main EFN + ArcFace model
    - EFN: Energy Flow Network (IRC-safe)
    - ArcFaceHead: Hyperspherical prototype classifier
    - TopTaggingLinear: EFN + Linear baseline
    - TopTaggingCosLinear: EFN + Cosine Linear baseline
"""

from .models.efn_arcface import TopTaggingModel, EFN, ArcFaceHead, arcface_loss, count_parameters
from .models.efn_linear import TopTaggingLinear
from .models.efn_coslinear import TopTaggingCosLinear

from .data.loader import load_awkward, load_events_by_indices, JetTaggingDataset, get_h5_len
from .utils.physics import compute_features, get_lund_coordinates
from .utils.clustering import estimate_clusters_eigengap, spectral_clustering_subclass
from .utils.device import get_device

__all__ = [
    'TopTaggingModel', 'EFN', 'ArcFaceHead',
    'TopTaggingLinear', 'TopTaggingCosLinear',
    'arcface_loss', 'count_parameters',
    'load_awkward', 'load_events_by_indices', 'JetTaggingDataset', 'get_h5_len',
    'compute_features', 'get_lund_coordinates',
    'estimate_clusters_eigengap', 'spectral_clustering_subclass',
    'get_device',
]
