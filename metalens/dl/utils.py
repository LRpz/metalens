import os
import albumentations as A
import cv2
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import timm
import torch
from albumentations.pytorch import ToTensorV2
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from scipy.spatial.distance import cdist
from skimage import measure, morphology
from sklearn.model_selection import train_test_split
from tifffile import imread
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp
from scipy.stats import zscore
from torchmetrics import PearsonCorrCoef, R2Score, Accuracy, F1Score
import math 

def define_transforms(patch_size=128):
    """
    Define data augmentation transforms for training and inference.
    
    Args:
        patch_size (int): Size of image patches
        
    Returns:
        list: [hard_transform, transform, post_transform] List of Albumentations transforms
    """
    hard_transform = A.Compose([
        A.RandomResizedCrop(height=patch_size, width=patch_size, scale=(1, 1), p=1)
        ])

    transform = A.Compose([
        A.Affine(scale=[0.75, 1], rotate=[-45, 45], p=0.5, mode=cv2.BORDER_REFLECT),
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Transpose(p=0.5),
        ])

    post_transform = A.Compose([ 
        ToTensorV2()
        ])

    return [hard_transform, transform, post_transform]

def process_annotations(annotations, lod_threshold=0.0, remove_zeros_samples=True):
    """
    Process and normalize metabolite annotations.
    
    Args:
        annotations (pd.DataFrame): DataFrame containing metabolite intensities and filenames
        lod_threshold (float): Limit of detection threshold
        remove_zeros_samples (bool): Whether to remove samples with all values below threshold
        
    Returns:
        tuple: (df_normalized, metabolites, weights) Normalized data, metabolite names, and weights
    """
    metabolites = np.array(annotations.drop('filename', axis=1).columns)

    df_normalized = annotations.copy()
    for col in metabolites:
        if annotations[col].dtype in ['float64', 'int64']:
            df_normalized.loc[:, col] = -10
            df_normalized.loc[annotations[col] > lod_threshold, col] = zscore(annotations.loc[annotations[col] > lod_threshold, col])

    if remove_zeros_samples: df_normalized = df_normalized.loc[df_normalized[metabolites].sum(axis=1) != -10*len(metabolites)]

    label_frequencies = df_normalized.drop('filename', axis=1).apply(lambda x: (x != -10).mean(), axis=0).values

    inv_freq = 1 - label_frequencies
    weights = (inv_freq - inv_freq.min()) / (inv_freq.max() - inv_freq.min())

    return df_normalized, metabolites, weights

class ImageDataset(Dataset):
    """
    Dataset class for loading and preprocessing microscopy images and metabolite data.
    
    Args:
        annotations (pd.DataFrame): DataFrame containing image filenames and metabolite values
        root_dir (str): Directory containing the image files
        metabolites (list): List of metabolite names
        hard_transform (callable, optional): Initial transformation
        transform (callable, optional): Data augmentation transformations
        post_transform (callable, optional): Final transformation (e.g., to tensor)
        task (str): Either 'classification' or 'regression'
    """
    def __init__(self, annotations, root_dir, metabolites, hard_transform=None, transform=None, post_transform=None, task='classification'):
        self.annotations = annotations
        self.root_dir = root_dir
        self.hard_transform = hard_transform
        self.transform = transform
        self.post_transform = post_transform
        self.metabolites = metabolites
        self.task = task

    def mask_central_am(self, image):
        """
        Mask all ablation marks except the central one.
        
        Args:
            image (np.ndarray): Input image with ablation mark channel
            
        Returns:
            np.ndarray: Image with only central ablation mark
        """
        binary_image = image[..., -1] > 0.5

        # Label the image
        labeled_image, num_labels = measure.label(binary_image, return_num=True)
        
        if num_labels > 1:
        
            image_center = np.array([[labeled_image.shape[0] / 2, labeled_image.shape[1] / 2]])
            properties = measure.regionprops(labeled_image)
            centroids = np.array([prop.centroid for prop in properties])
            distances = cdist(centroids, image_center)
            centermost_label = properties[np.argmin(distances)].label
            centermost_mask = labeled_image == centermost_label
            dilated_mask = morphology.dilation(centermost_mask, morphology.disk(3))
            image[..., -1] = image[..., -1] * dilated_mask

        return image

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations.loc[idx, 'filename'])
        image = imread(img_name)
        
        # Experiment - skip BF of AM -- Final solution
        # image = image[..., [0, 1, 2, 4]]


        image = np.moveaxis(image, -1, 0)
        image = image.transpose((1, 2, 0))  # Convert to HWC format for Albumentations

        metabolite_intensities = self.annotations.loc[idx, self.metabolites].values.astype(np.float32)
        if self.task == 'regression':
            label = torch.from_numpy(metabolite_intensities).float() # Regression
        elif self.task == 'classification':
            label = torch.from_numpy(metabolite_intensities > 0) # Classification

        if self.hard_transform:
            image = self.hard_transform(image=image)['image']

        if self.transform:
            image = self.transform(image=image)['image']
            image = self.mask_central_am(image)

        if self.post_transform:
            image = self.post_transform(image=image)['image']
        
        return image, label

    def __len__(self):
        return len(self.annotations)

class ImageRegressor(pl.LightningModule):
    """
    PyTorch Lightning module for metabolite prediction from microscopy images.
    
    Args:
        num_classes (int): Number of metabolites to predict
        metabolite_weights (bool or np.ndarray): Weights for metabolite loss calculation
        learning_rate (float): Initial learning rate
        n_epochs (int): Number of training epochs
        encoder (str): Name of the encoder architecture ('resnet152' or 'mit_b5')
        in_chans (int): Number of input channels
    """
    def __init__(self, num_classes, metabolite_weights=False, learning_rate=1e-3, n_epochs=200, encoder='resnet152', in_chans=4):
        super().__init__()

        # Configure encoder based on architecture
        if encoder == 'mit_b5': 
            weights = 'imagenet'
            in_channels = 3
        else: 
            weights = None
            in_channels = in_chans

        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder, 
            encoder_weights=weights, 
            in_channels=in_channels, 
            classes=num_classes,
        )
        
        if encoder == 'mit_b5': 
            self.adapt_input_model(in_chans)

        self.learning_rate = learning_rate
        self.epochs = n_epochs
        self.weights = metabolite_weights

        # Initialize metrics
        self.pearson_corrcoef = PearsonCorrCoef(num_outputs=1)
        self.r2_score = R2Score(num_outputs=1)

    def adapt_input_conv(self, in_chans, conv_weight):
        """
        This function adapts the input channels of a convolutional layer's weights based on the number of input channels 
        provided. It handles cases where the input channels are 1 (grayscale), 3 (RGB), or other values. 
        The function ensures that the weight tensor is in the correct format for the given number of input channels and 
        adjusts the weights accordingly.

        Args:
        in_chans (int): The number of input channels.
        conv_weight (torch.Tensor): The convolutional layer's weights.

        Returns:
        torch.Tensor: The adapted convolutional layer's weights.
        """
        conv_type = conv_weight.dtype
        conv_weight = conv_weight.float()  # Some weights are in torch.half, ensure it's float for sum on CPU
        O, I, J, K = conv_weight.shape
        if in_chans == 1:
            if I > 3:
                assert conv_weight.shape[1] % 3 == 0
                # For models with space2depth stems
                conv_weight = conv_weight.reshape(O, I // 3, 3, J, K)
                conv_weight = conv_weight.sum(dim=2, keepdim=False)
            else:
                conv_weight = conv_weight.sum(dim=1, keepdim=True)
        elif in_chans != 3:
            if I != 3:
                raise NotImplementedError('Weight format not supported by conversion.')
            else:
                # NOTE this strategy should be better than random init, but there could be other combinations of
                # the original RGB input layer weights that'd work better for specific cases.
                repeat = int(math.ceil(in_chans / 3))
                conv_weight = conv_weight.repeat(1, repeat, 1, 1)[:, :in_chans, :, :]
                conv_weight *= (3 / float(in_chans))
        conv_weight = conv_weight.to(conv_type)

        return conv_weight

    def adapt_input_model(self, in_channels):
        """
        Adapts first layer to take a specified number of input channels.
        
        Args:
            model: The segmentation model to be adapted.

        Returns:
            The adapted segmentation model.
        """
        # Adapt first layer to take 1 channel as input - timm approach = sum weights
        new_weights = self.adapt_input_conv(in_chans=in_channels, conv_weight=self.model.encoder.patch_embed1.proj.weight)
        self.model.encoder.patch_embed1.proj = torch.nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(7, 7), stride=(4, 4), padding=(3, 3))

        with torch.no_grad():
            self.model.encoder.patch_embed1.proj.weight = torch.nn.parameter.Parameter(new_weights)
        
        # return model

    def custom_weighted_mse_loss(self, y_hat, y, weights, non_zero_mask):
        """
        Calculate weighted MSE loss for metabolite predictions.
        
        Args:
            y_hat (torch.Tensor): Predicted values
            y (torch.Tensor): Ground truth values
            weights (torch.Tensor): Weights for each metabolite
            non_zero_mask (torch.Tensor): Mask for non-zero values
            
        Returns:
            torch.Tensor: Weighted MSE loss
        """
        effective_loss = weights + (y_hat - y) ** 2 * non_zero_mask
        loss = effective_loss.sum() / non_zero_mask.sum()
        return loss

    def forward(self, x):
        """Forward pass through the model"""
        x = self.model(x)
        return x

    def compute_loss(self, image, y, non_zero_mask):
        """
        Compute loss and predictions for a batch.
        
        Args:
            image (torch.Tensor): Input image batch
            y (torch.Tensor): Ground truth values
            non_zero_mask (torch.Tensor): Mask for non-zero values
            
        Returns:
            tuple: (loss, predictions)
        """
        image_pred = self(image).squeeze()

        mask = image[:, -1, ...].unsqueeze(1)
        masked_preds = image_pred * mask
        y_hat = torch.sum(masked_preds.view(masked_preds.size(0), masked_preds.size(1), -1), dim=-1)

        loss = self.custom_weighted_mse_loss(
            y_hat,
            y, 
            torch.from_numpy(self.weights).to('cuda'), 
            non_zero_mask
        )

        return loss, y_hat

    def training_step(self, batch):
        """Training step for one batch"""
        image, y = batch
        non_zero_mask = y != -10 

        loss, y_hat = self.compute_loss(image, y, non_zero_mask)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_pearson_corrcoef', 
                self.pearson_corrcoef(y_hat[non_zero_mask], y[non_zero_mask]).mean(), 
                on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_r2_score', 
                self.r2_score(y_hat[non_zero_mask], y[non_zero_mask]).mean(), 
                on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def validation_step(self, batch):
        """Validation step for one batch"""
        image, y = batch
        non_zero_mask = y != -10 

        loss, y_hat = self.compute_loss(image, y, non_zero_mask)
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True, logger=True)
        self.log('val_pearson_corrcoef', 
                self.pearson_corrcoef(y_hat[non_zero_mask], y[non_zero_mask]).mean(), 
                on_epoch=True, prog_bar=True, logger=True)
        self.log('val_r2_score', 
                self.r2_score(y_hat[non_zero_mask], y[non_zero_mask]).mean(), 
                on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)

        scheduler = {
            'scheduler': StepLR(optimizer, step_size=self.epochs//4, gamma=0.5),
            'monitor': 'val_loss',
            'name': 'step_lr'
        }

        return [optimizer], [scheduler]

def get_loaders(folder_path, annotations, metabolites, batch_size, train_annotations, val_annotations, hard_transform, transform, post_transform, task, num_workers=2):
    """
    Create data loaders for training and validation.
    
    Args:
        folder_path (str): Path to data directory
        annotations (pd.DataFrame): Full annotations DataFrame
        metabolites (list): List of metabolite names
        batch_size (int): Batch size for training
        train_annotations (pd.DataFrame): Training set annotations
        val_annotations (pd.DataFrame): Validation set annotations
        hard_transform (callable): Initial transformation
        transform (callable): Data augmentation transformations
        post_transform (callable): Final transformation
        task (str): Either 'classification' or 'regression'
        num_workers (int): Number of data loading workers
        
    Returns:
        tuple: (train_dataloader, val_dataloader, annotations)
    """
    train_dataset = ImageDataset(
        annotations=train_annotations.reset_index(), 
        metabolites=metabolites, 
        root_dir=folder_path, 
        hard_transform=hard_transform,
        transform=transform, 
        post_transform=post_transform,
        task=task
    )

    persistent = num_workers > 0

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        pin_memory=True, 
        num_workers=num_workers,
        persistent_workers=persistent,
        drop_last=True,
    )

    val_dataset = ImageDataset(
        val_annotations.reset_index(), 
        metabolites=metabolites, 
        root_dir=folder_path, 
        hard_transform=hard_transform,
        transform=None, 
        post_transform=post_transform,
        task=task
    )    
        
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        pin_memory=True, 
        num_workers=num_workers, 
        persistent_workers=persistent,
        drop_last=True
    )

    return train_dataloader, val_dataloader, annotations

def train_regressor(folder_path, model_path, batch_size, learning_rate, epochs, encoder, transform_collection, in_chans, test_size=0.33, metabolite_oi=None):
    """
    Train a metabolite prediction model.
    
    Args:
        folder_path (str): Path to training data directory
        model_path (str): Path to save model checkpoints
        batch_size (int): Training batch size
        learning_rate (float): Initial learning rate
        epochs (int): Number of training epochs
        encoder (str): Name of encoder architecture
        transform_collection (list): List of data transformations
        in_chans (int): Number of input channels
        test_size (float): Fraction of data to use for validation
        metabolite_oi (list, optional): List of specific metabolites to predict
        
    Returns:
        tuple: (model, trainer) Trained model and PyTorch Lightning trainer
    """
    # Load and process annotations
    annotations = pd.read_csv(os.path.join(folder_path, 'ion_intensities.csv'))
    df_normalized, metabolites, weights = process_annotations(annotations, remove_zeros_samples=True)

    if metabolite_oi is not None: 
        metabolites = metabolite_oi

    # Split data into train and validation sets
    train_annotations, val_annotations = train_test_split(
        df_normalized, 
        test_size=test_size, 
        random_state=42
    )

    hard_transform, transform, post_transform = transform_collection

    # Create data loaders
    train_dataloader, val_dataloader, annotations = get_loaders(
        folder_path=folder_path, 
        annotations=annotations, 
        metabolites=metabolites, 
        batch_size=batch_size, 
        train_annotations=train_annotations, 
        val_annotations=val_annotations, 
        hard_transform=hard_transform, 
        transform=transform, 
        post_transform=post_transform,
        task='regression', 
        num_workers=2
    )

    # Initialize model
    model = ImageRegressor(
        num_classes=len(metabolites), 
        metabolite_weights=weights, 
        n_epochs=epochs, 
        encoder=encoder, 
        learning_rate=learning_rate, 
        in_chans=in_chans
    )
    
    # Set up logging and checkpointing
    logger_dirname = f'{encoder}_lr_{learning_rate}_bs_{batch_size}_epochs_{epochs}'
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_pearson_corrcoef',
        dirpath=os.path.join(model_path, logger_dirname), 
        filename='{epoch:02d}-{val_loss:.4f}-{val_pearson_corrcoef:.3f}-{val_r2_score:.3f}',
        save_top_k=1, 
        mode='max'
    )

    logger = TensorBoardLogger(
        save_dir=model_path, 
        name=logger_dirname
    )
    
    # Train model
    trainer = pl.Trainer(
        max_epochs=epochs, 
        callbacks=[checkpoint_callback], 
        logger=logger
    )
    trainer.fit(model, train_dataloader, val_dataloader)

    return model, trainer

def load_regressor(checkpoint_path, num_classes, encoder, in_chans):
    """
    Load a trained regressor model from checkpoint.
    
    Args:
        checkpoint_path (str): Path to model checkpoint file
        num_classes (int): Number of metabolites to predict
        encoder (str): Name of encoder architecture
        in_chans (int): Number of input channels
        
    Returns:
        ImageRegressor: Loaded model in evaluation mode
    """
    model = ImageRegressor.load_from_checkpoint(
        checkpoint_path,
        num_classes=num_classes,
        encoder=encoder,
        in_chans=in_chans
    )
    model.eval()
    return model
