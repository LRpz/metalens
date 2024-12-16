import glob
import numpy as np
import tifffile as tif
import torch
import segmentation_models_pytorch as smp
from monai.inferers import SlidingWindowInferer
import math
import sys
import os

def adapt_input_conv(in_chans, conv_weight):
    """
    Adapt convolutional layer weights for different input channels.
    
    Args:
        in_chans (int): Number of input channels
        conv_weight (torch.Tensor): Original convolution weights
        
    Returns:
        torch.Tensor: Adapted convolution weights
    """
    conv_type = conv_weight.dtype
    conv_weight = conv_weight.float()
    O, I, J, K = conv_weight.shape
    
    if in_chans == 1:
        if I > 3:
            assert conv_weight.shape[1] % 3 == 0
            conv_weight = conv_weight.reshape(O, I // 3, 3, J, K)
            conv_weight = conv_weight.sum(dim=2, keepdim=False)
        else:
            conv_weight = conv_weight.sum(dim=1, keepdim=True)
    elif in_chans != 3:
        if I != 3:
            raise NotImplementedError('Weight format not supported by conversion.')
        else:
            repeat = int(math.ceil(in_chans / 3))
            conv_weight = conv_weight.repeat(1, repeat, 1, 1)[:, :in_chans, :, :]
            conv_weight *= (3 / float(in_chans))
            
    return conv_weight.to(conv_type)

def adapt_input_model(model):
    """
    Adapt model's first layer to accept single-channel input.
    
    Args:
        model (torch.nn.Module): Original model
        
    Returns:
        torch.nn.Module: Adapted model
    """
    new_weights = adapt_input_conv(in_chans=1, conv_weight=model.encoder.patch_embed1.proj.weight)
    model.encoder.patch_embed1.proj = torch.nn.Conv2d(
        in_channels=1, 
        out_channels=64, 
        kernel_size=(7, 7), 
        stride=(4, 4), 
        padding=(3, 3)
    )

    with torch.no_grad():
        model.encoder.patch_embed1.proj.weight = torch.nn.parameter.Parameter(new_weights)
    
    return model

def scale(arr):
    """Min-max scale array to [0,1] range"""
    arr_min = arr.min()
    arr_max = arr.max()
    return (arr - arr_min) / (arr_max - arr_min)

if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: AM_segmenation_inference.py <dataset_name>")
        sys.exit(1)
    
    ds_name = sys.argv[1]

    # Setup paths
    data_dir = os.path.join('data', 'raw_data')
    input_path = os.path.join(data_dir, f'{ds_name}_ablation_marks_tf.tif')
    model_path = os.path.join('models', 'AM_segmentation.pth')

    device='cuda'

    # Initialize model
    model = smp.Unet(
        encoder_name='mit_b5', 
        classes=1, 
        in_channels=3, 
        encoder_weights=None
    )
    model = adapt_input_model(model)
    
    # Load model weights
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)

    # Configure sliding window inference
    inferer = SlidingWindowInferer(
        roi_size=(1024, 1024), 
        sw_batch_size=1, 
        progress=True, 
        mode="gaussian",
        overlap=0.5,
        device='cpu',    # Stitching on CPU
        sw_device=device # Inference on GPU
    )

    # Load and preprocess input image
    im_np = scale(tif.imread(input_path))
    im_tensor = torch.tensor(im_np[None, None, ...]).float().to(device)

    # Run inference
    with torch.no_grad():
        pred_tensor = inferer(inputs=im_tensor, network=model)
        pred_tensor = torch.sigmoid(pred_tensor)

    # Save predictions
    pred = pred_tensor.squeeze().cpu().numpy()
    output_path = input_path.replace('.tif', '_pred.tif')
    tif.imwrite(output_path, pred.astype(np.float32))
    print(f"Predictions saved to: {output_path}")