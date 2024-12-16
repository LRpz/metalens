import torch
import os
import sys
import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import tifffile as tif
import numpy as np
import matplotlib.pyplot as plt
import h5py
from metalens.dl.utils import process_annotations, define_transforms, load_regressor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collect_annotation_stats(annotations, metabolites):
    """
    Calculate mean and standard deviation for each metabolite's non-zero values.
    
    Args:
        annotations (pd.DataFrame): DataFrame containing metabolite intensities
        metabolites (list): List of metabolite names
        
    Returns:
        tuple: (stds, means) Lists of standard deviations and means for each metabolite
    """
    means = []
    stds = []
    for metabolite in metabolites:
        data = annotations[metabolite].values
        data = data[data != 0]
        means.append(data.mean())
        stds.append(data.std())
    return stds, means

def scale_cuda(patch):
    """
    Min-max scale a tensor to [0,1] range.
    
    Args:
        patch (torch.Tensor): Input tensor
        
    Returns:
        torch.Tensor: Scaled tensor
    """
    min_val = patch.min()
    max_val = patch.max()
    return (patch - min_val) / (max_val - min_val)

def contrast_cuda(patch, min_percentile, max_percentile):
    """
    Apply contrast enhancement using percentile-based normalization.
    
    Args:
        patch (torch.Tensor): Input tensor
        min_percentile (float): Lower percentile for normalization
        max_percentile (float): Upper percentile for normalization
        
    Returns:
        torch.Tensor: Contrast-enhanced tensor
    """
    min_val = torch.quantile(patch, min_percentile / 100.0)
    max_val = torch.quantile(patch, max_percentile / 100.0)
    return torch.clamp((patch - min_val) / (max_val - min_val), 0.0, 1.0)

def load_annotations(folder_path, split=True, test_size=0.3):
    """
    Load and process metabolite annotations from CSV file.
    
    Args:
        folder_path (str): Path to folder containing ion_intensities.csv
        split (bool): Whether to split data into train/val sets
        test_size (float): Fraction of data to use for validation
        
    Returns:
        tuple: (metabolites, val_annotations, annot_stats)
    """
    annotations = pd.read_csv(os.path.join(folder_path, 'ion_intensities.csv'))
    df_normalized, metabolites, weights = process_annotations(annotations)    
    if split:
        train_annotations, val_annotations = train_test_split(annotations, test_size=test_size, random_state=42)
    else:
        val_annotations = annotations
    annot_stats = collect_annotation_stats(annotations, metabolites)
    return metabolites, val_annotations, annot_stats

def pred_images(test_image, model, am_test, annot_stats, metabolites, start_x=0, start_y=0, eval_range=500, step=1, in_chans=4, patch_size=128, batch_size=128, progress_callback=None):
    """
    Run inference on microscopy image patches.
    
    Args:
        test_image (np.ndarray): Input microscopy image
        model (torch.nn.Module): Trained model
        am_test (np.ndarray): Ablation mark mask
        annot_stats (tuple): Statistics from annotations
        metabolites (list): List of metabolite names
        start_x (int): Starting X coordinate for inference
        start_y (int): Starting Y coordinate for inference
        eval_range (int): Size of region to evaluate
        step (int): Stride between patches
        in_chans (int): Number of input channels
        patch_size (int): Size of image patches
        batch_size (int): Batch size for inference
        progress_callback (callable): Function to call for progress updates
        
    Returns:
        tuple: (pred_image_r_cpu, pred_image_counts_cpu) Predicted metabolite maps and count maps
    """
    def process_patches(patches, model):
        """Process a batch of patches through the model"""
        with torch.no_grad():
            output = model(patches)
        output_norm = output * mask
        return output_norm
    
    # Prepare input data
    if len(test_image.shape) == 2: 
        test_image = test_image[..., None]
    test_image_tensor = torch.from_numpy(np.moveaxis(test_image, -1, 0)).unsqueeze(0).to(device)
    
    # Initialize tensors
    patches = torch.empty(0, in_chans, patch_size, patch_size, device='cuda')
    coords = []
    
    # Prepare statistics and mask
    stds, means = annot_stats
    mean_tensor = torch.tensor(means).view(1, len(metabolites), 1, 1).to(device)
    std_tensor = torch.tensor(stds).view(1, len(metabolites), 1, 1).to(device)    
    
    am_image = torch.from_numpy(np.moveaxis(am_test, -1, 0)).to(device).unsqueeze(0)
    mask = am_image[0, ...] > 0.5
    mask = mask.to(torch.float32)
    
    # Initialize output tensors
    pred_image_r = torch.zeros((1, len(metabolites), eval_range+patch_size, eval_range+patch_size), device='cuda')
    pred_image_counts = torch.zeros((eval_range+patch_size, eval_range+patch_size), device='cuda')
    
    # Generate evaluation coordinates
    x_eval = np.arange(start_x, start_x+eval_range, step)
    y_eval = np.arange(start_y, start_y+eval_range, step)
    
    # Main inference loop
    for i in tqdm.tqdm(x_eval):
        for j in y_eval:
            # Extract and process patch
            patch = test_image_tensor[:, :, i:i+patch_size, j:j+patch_size]
            if patch.shape[1] == 1:
                cell_patch = scale_cuda(contrast_cuda(patch, 0.1, 99.9))
            else:
                cell_patch = torch.stack([scale_cuda(contrast_cuda(patch[0, chan, ...], 0.1, 99.9)) 
                                        for chan in range(patch.shape[1])], dim=0).unsqueeze(0)
            
            patch_data = torch.cat([cell_patch, am_image[None, ...]], dim=1)
            patches = torch.cat([patches, patch_data], dim=0)
            coords.append((i, j))
            
            # Process batch if full
            if patches.shape[0] == batch_size:
                output = process_patches(patches, model)
                for k, (x, y) in enumerate(coords):
                    x, y = x-start_x, y-start_y
                    pred_image_r[..., x:x+patch_size, y:y+patch_size] += output[k]
                    pred_image_counts[x:x+patch_size, y:y+patch_size] += patches[k, -1, ...] > 0.5
                
                if progress_callback:
                    progress_callback()
                
                patches = torch.empty(0, in_chans, patch_size, patch_size, device='cuda')
                coords = []
    
    # Process remaining patches
    if patches.shape[0] > 0:
        output = process_patches(patches, model)
        for k, (x, y) in enumerate(coords):
            x, y = x-start_x, y-start_y
            pred_image_r[..., x:x+patch_size, y:y+patch_size] += output[k]
            pred_image_counts[x:x+patch_size, y:y+patch_size] += patches[k, -1, ...] > 0.5
    
    # Convert to CPU and correct format
    pred_image_r_cpu = np.moveaxis(pred_image_r.cpu().numpy()[0], 0, -1)
    pred_image_counts_cpu = pred_image_counts.cpu().numpy()
    
    return pred_image_r_cpu, pred_image_counts_cpu

def save_data(data, h5_filepath):
    """
    Save prediction data to HDF5 file.
    
    Args:
        data (dict): Dictionary containing 'pred' and 'metabolites' arrays
        h5_filepath (str): Path to save the HDF5 file
    """
    with h5py.File(h5_filepath, 'w') as file:
        for key, value in data.items():
            # Create a dataset for each item
            file.create_dataset(key, data=np.array(value))

plt.style.use('default')

if __name__ == "__main__":
    """
    Command-line interface for running inference on microscopy images.
    
    This script loads a trained model and runs inference on a single microscopy image,
    predicting metabolite concentrations for each ablation mark.
    
    Usage:
        python eval.py <dataset_name> <model_path>
    
    Example:
        python eval.py sample_001 models/checkpoints/best_model.ckpt
    """
    # Parse command line arguments
    if len(sys.argv) < 3:
        print("Usage: eval.py <dataset_name> <model_path>")
        sys.exit(1)
    
    sample_id = sys.argv[1]
    checkpoint_path = sys.argv[2]

    # Model configuration
    in_chans = 4
    patch_size = 128
    encoder = 'resnet152'

    # Load annotations and statistics
    folder_path = os.path.join('data', 'training_data')
    metabolites, val_annotations, annot_stats = load_annotations(os.path.join(folder_path))

    # Load and preprocess input data
    test_image = tif.imread(os.path.join('data', 'raw_data', f'{sample_id}_cells.tif'))
    # Correct for microscope alignment
    test_image[..., 2] = np.roll(test_image[..., 2], shift=3, axis=0)  # Shift down by 3 pixels
    test_image[:3, :, 2] = np.median(test_image[..., 2])  # Fill the top 3 rows with the median value

    # Load ablation mark data
    am_test = tif.imread(os.path.join('data', 'am_eval.tif'))[..., -1]  # Only take the last channel (AM probability map)

    # Initialize model
    model = load_regressor(
        checkpoint_path=checkpoint_path,
        num_classes=len(metabolites),
        encoder=encoder,
        in_chans=in_chans
    )

    # Run inference
    eval_range = 500
    pred_image_r_cpu, pred_image_counts_cpu = pred_images(
        test_image=test_image,
        model=model, 
        am_test=am_test,
        start_x=0, 
        start_y=0, 
        eval_range=eval_range, 
        step=4,                    # Stride between predictions
        annot_stats=annot_stats,
        metabolites=metabolites, 
        batch_size=128             # Adjust based on GPU memory
    )

    # Prepare results for saving
    data = {
        'pred': pred_image_r_cpu,  
        'metabolites': metabolites
    }

    # Save predictions
    output_dir = os.path.join('data', 'output')
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
    h5_filepath = os.path.join(output_dir, 'output.h5')
    save_data(data, h5_filepath)
    print(f"Results saved to: {h5_filepath}")