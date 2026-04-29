"""
Deep learning sub-package for MetaLens.

Public API
----------
pred_images
    Run sliding-window inference on a microscopy image.
load_annotations
    Load and split metabolite annotation CSV for evaluation.
load_regressor
    Load a trained ImageRegressor from a Lightning checkpoint.
define_transforms
    Build the Albumentations transform pipeline for training/inference.
"""
from .eval import pred_images, load_annotations
from .utils import load_regressor, define_transforms

__all__ = [
    'pred_images',
    'load_annotations',
    'load_regressor',
    'define_transforms'
] 