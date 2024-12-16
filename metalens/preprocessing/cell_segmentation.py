"""
Cell Segmentation Script

This script performs cell segmentation on microscopy images using the Cellpose model.
It processes multi-channel microscopy images and generates cell masks.

Usage:
    python cell_segmentation.py <dataset_name>

Example:
    python cell_segmentation.py sample_001
"""
import numpy as np
import tifffile as tif
import sys
import os
from cellpose import models
from cellpose.io import imread

def scale(arr):
    """
    Min-max scale array to [0,1] range.
    
    Args:
        arr (np.ndarray): Input array
        
    Returns:
        np.ndarray: Scaled array
    """
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

def contrast(arr, low, high):
    """
    Apply contrast enhancement using percentile clipping.
    
    Args:
        arr (np.ndarray): Input array
        low (float): Lower percentile threshold
        high (float): Upper percentile threshold
        
    Returns:
        np.ndarray: Contrast-enhanced array
    """
    return np.clip(arr, np.percentile(arr, low), np.percentile(arr, high))

if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python cell_segmentation.py <dataset_name>")
        sys.exit(1)
    
    ds_name = sys.argv[1]
    
    # Setup paths
    data_dir = os.path.join('data', 'raw_data')
    input_path = os.path.join(data_dir, f'{ds_name}_cells.tif')
    
    # Load input image
    imgs = [imread(input_path)]
    
    # Configure Cellpose
    channels = [[2, 3]]  # Using channels 2 and 3 for segmentation
    model = models.Cellpose(model_type='cyto2', gpu=True)
    
    # Run cell segmentation
    print(f"Running Cellpose segmentation on {input_path}")
    masks, flows, styles, diams = model.eval(
        imgs, 
        diameter=50, 
        channels=channels,
        flow_threshold=1)
    
    # Save segmentation masks
    output_path = input_path.replace('cells', 'cells_mask')
    tif.imwrite(output_path, masks[0].astype(np.uint16))
    print(f"Cell masks saved to: {output_path}")
