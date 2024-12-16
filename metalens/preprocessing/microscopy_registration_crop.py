"""
Microscopy Image Registration and Cropping Script

This script performs registration and cropping of pre- and post-MALDI microscopy images.
It applies affine transformations for alignment and crops the images to the region
containing ablation marks.

Usage:
    python microscopy_registration_crop.py <dataset_name>

Example:
    python microscopy_registration_crop.py sample_001
"""
import numpy as np
import tifffile as tif
from skimage import transform, io
from pystackreg import StackReg
import glob
import os
import tqdm
import sys

def scale(im):
    """
    Min-max scale array to [0,1] range.
    
    Args:
        im (np.ndarray): Input array
        
    Returns:
        np.ndarray: Scaled array
    """
    return (im - np.min(im)) / (np.max(im) - np.min(im))

def contrast(im, min_percentile, max_percentile):
    """
    Apply contrast enhancement using percentile clipping.
    
    Args:
        im (np.ndarray): Input array
        min_percentile (float): Lower percentile threshold
        max_percentile (float): Upper percentile threshold
        
    Returns:
        np.ndarray: Contrast-enhanced array
    """
    min_val = np.percentile(im, min_percentile)
    max_val = np.percentile(im, max_percentile)
    return np.clip(im, min_val, max_val)

def process_images(base_path, ds_name):
    """
    Process and align microscopy images.
    
    Args:
        base_path (str): Path to data directory
        ds_name (str): Dataset name
        
    Returns:
        tuple: (crop_pre, crop_post) Pre- and post-MALDI cropped images
    """
    # Setup paths
    im_pre_path = os.path.join(base_path, f"{ds_name}_preMALDI_channel1")
    im_post_path = os.path.join(base_path, f"{ds_name}_postMALDI_channel1")
    data_path = os.path.join(base_path, ds_name, "transformedMarks.npy")
    params_path = os.path.join(base_path, ds_name, "optimized_params.npy")

    # Load input data
    im_pre = tif.imread(im_pre_path)
    im_post = tif.imread(im_post_path)
    data = np.load(data_path)
    tx, ty, angle = np.load(params_path)

    # Apply initial affine transformation
    affine_transform = transform.SimilarityTransform(
        translation=(ty, tx), 
        rotation=-angle
    )
    im_post_tf = transform.warp(im_post, affine_transform.inverse) 

    # Fine-tune alignment using StackReg
    scaling = 1  # Optional scaling factor
    crop_shape = np.min([im_pre.shape, im_post.shape], axis=0)
    
    ref = transform.resize(
        im_pre[:crop_shape[0], :crop_shape[1]], 
        crop_shape // scaling, 
        anti_aliasing=True
    )
    mov = transform.resize(
        im_post_tf[:crop_shape[0], :crop_shape[1]], 
        crop_shape // scaling, 
        anti_aliasing=True
    )

    sr = StackReg(StackReg.RIGID_BODY)
    out = sr.register_transform(ref, mov)

    # Calculate crop bounds around ablation marks
    window = 100  # Margin around ablation marks
    min_y, min_x = np.min(data, axis=1) - window
    max_y, max_x = np.max(data, axis=1) + window
    min_x, min_y, max_x, max_y = [max(0, int(coord)) for coord in [min_x, min_y, max_x, max_y]]

    # Crop and process images
    crop_post = scale(out[min_y:max_y, min_x:max_x])

    # Load and process all pre-MALDI channels
    file_list = sorted(glob.glob(os.path.join(base_path, f"{ds_name}_preMALDI_*")))
    
    if not file_list:
        raise FileNotFoundError(f"No pre-MALDI files found for dataset: {ds_name}")

    channels = [
        scale(tif.imread(file)[min_y:max_y, min_x:max_x]) 
        for file in file_list
    ]
    
    crop_pre = np.stack(channels, axis=0)

    return crop_pre, crop_post

if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python microscopy_registration_crop.py <dataset_name>")
        sys.exit(1)
    
    ds_name = sys.argv[1]

    # Setup paths
    root = os.path.join('data', 'raw_data')
    
    # Process images
    print(f"Processing dataset: {ds_name}")
    crop_pre, crop_post = process_images(root, ds_name)

    # Save processed images
    output_pre = os.path.join(root, f'{ds_name}_cells.tif')
    output_post = os.path.join(root, f'{ds_name}_ablation_marks_tf.tif')
    
    tif.imwrite(
        output_pre,
        crop_pre.astype(np.float32)
    )
    tif.imwrite(
        output_post,
        crop_post.astype(np.float32)
    )
    print(f"Saved processed images to:\n  {output_pre}\n  {output_post}")