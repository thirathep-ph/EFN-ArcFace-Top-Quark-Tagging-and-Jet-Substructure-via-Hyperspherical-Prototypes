"""
arcefn model definitions.
"""
from .efn_arcface import TopTaggingModel, EFN, ArcFaceHead, arcface_loss, count_parameters
from .efn_linear import TopTaggingLinear
from .efn_coslinear import TopTaggingCosLinear

__all__ = [
    'TopTaggingModel', 'EFN', 'ArcFaceHead',
    'TopTaggingLinear', 'TopTaggingCosLinear',
    'arcface_loss', 'count_parameters',
]
