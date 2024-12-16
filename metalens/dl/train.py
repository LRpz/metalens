"""
Training script for MetaLens metabolite prediction model.

This script provides a command-line interface for training the MetaLens model
on microscopy data to predict metabolite concentrations.

Usage:
    python train.py <training_data_folder> <model_folder>

Example:
    python train.py data/training_data models/checkpoints
"""
import sys
from metalens.dl.utils import define_transforms, train_regressor

if __name__ == '__main__':

    # Parse command line arguments
    if len(sys.argv) != 3:
        print("Usage: train.py <training_data_folder> <model_folder>")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    model_path = sys.argv[2]

    # Model configuration
    MODEL = 'resnet152'
    patch_size = 128
    training_patch_size = 128  # Patch size for training (used in data augmentation)
    
    # Training hyperparameters
    batch_size = 32
    learning_rate = 1e-3
    epochs = 200
    encoder = MODEL
    in_chans = 4 
    test_size = 0.3

    # Define data transformations
    transforms = define_transforms(training_patch_size)

    # Train model
    model, trainer = train_regressor(
         folder_path=folder_path,          # Path to training data
         model_path=model_path,            # Path to save model checkpoints
         batch_size=batch_size,            # 32 is more stable than 64
         learning_rate=learning_rate,      # 1e-3 gives good results
         epochs=epochs,                    # Total training epochs
         encoder=encoder,                  # Backbone architecture
         transform_collection=transforms,   # Data augmentation pipeline
         in_chans=in_chans,               # Number of input channels
         test_size=test_size              # Validation set fraction
         )