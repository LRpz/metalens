import napari
import numpy as np
from napari.layers import Image
from magicgui import magic_factory
import h5py
from pathlib import Path
import tifffile as tif
from typing import List
import torch
from qtpy.QtWidgets import (QVBoxLayout, QPushButton, QWidget, QFileDialog, 
                            QSpinBox, QLabel, QGridLayout, QGroupBox, QDoubleSpinBox)
from metalens.dl.utils import load_regressor, define_transforms
from metalens.dl.eval import pred_images, load_annotations
from napari_plugin_engine import napari_hook_implementation
from scipy.ndimage import gaussian_filter
from napari.utils import progress
from napari.utils.notifications import show_info

class MetaLensWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        # Show activity dock using window menu
        try:
            # Try to show activity dock through window actions
            for action in self.viewer.window.window_menu.actions():
                if 'Activity' in action.text():
                    action.setChecked(True)
                    action.trigger()
                    break
        except Exception as e:
            print(f"Could not automatically show activity dock: {e}")
        
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)
        
        # Add buttons
        self.load_microscopy_btn = QPushButton("Load Microscopy Data")
        self.load_model_btn = QPushButton("Load Model")
        self.run_inference_btn = QPushButton("Run Inference")
        self.load_predictions_btn = QPushButton("Load Predictions")
        self.save_results_btn = QPushButton("Save Results")
        self.save_results_btn.setEnabled(False)
        
        # Add buttons to layout
        main_layout.addWidget(self.load_microscopy_btn)
        main_layout.addWidget(self.load_model_btn)
        
        # Create inference parameters group
        inference_group = QGroupBox("Inference Parameters")
        inference_layout = QGridLayout()
        inference_group.setLayout(inference_layout)
        
        # Evaluation range input
        self.eval_range = QSpinBox()
        self.eval_range.setRange(100, 10000)
        self.eval_range.setValue(500)
        eval_range_label = QLabel("Evaluation Range (pixels):")
        eval_range_desc = QLabel("Size of the region to analyze (smaller = faster)")
        inference_layout.addWidget(eval_range_label, 0, 0)
        inference_layout.addWidget(self.eval_range, 0, 1)
        inference_layout.addWidget(eval_range_desc, 1, 0, 1, 2)
        
        # Step size input
        self.step_size = QSpinBox()
        self.step_size.setRange(1, 100)
        self.step_size.setValue(2)
        step_size_label = QLabel("Step Size:")
        step_size_desc = QLabel("Pixels to skip between predictions (larger = faster but coarser)")
        inference_layout.addWidget(step_size_label, 2, 0)
        inference_layout.addWidget(self.step_size, 2, 1)
        inference_layout.addWidget(step_size_desc, 3, 0, 1, 2)
        
        # Batch size input
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 1024)
        self.batch_size.setValue(128)
        batch_size_label = QLabel("Batch Size:")
        batch_size_desc = QLabel("Number of patches to process at once (larger = faster but more memory)")
        inference_layout.addWidget(batch_size_label, 4, 0)
        inference_layout.addWidget(self.batch_size, 4, 1)
        inference_layout.addWidget(batch_size_desc, 5, 0, 1, 2)
        
        # Add inference group to main layout
        main_layout.addWidget(inference_group)
        
        # Create display parameters group
        display_group = QGroupBox("Display Parameters")
        display_layout = QGridLayout()
        display_group.setLayout(display_layout)
        
        # Gaussian blur sigma input
        self.sigma = QDoubleSpinBox()
        self.sigma.setRange(0, 10)        
        self.sigma.setValue(0)
        self.sigma.setSingleStep(0.1)
        sigma_label = QLabel("Gaussian Blur σ:")
        sigma_desc = QLabel("Smoothing factor (0 = no blur)")
        display_layout.addWidget(sigma_label, 0, 0)
        display_layout.addWidget(self.sigma, 0, 1)
        display_layout.addWidget(sigma_desc, 1, 0, 1, 2)
        
        # Add display group to main layout
        main_layout.addWidget(display_group)
        
        main_layout.addWidget(self.run_inference_btn)
        main_layout.addWidget(self.load_predictions_btn)
        main_layout.addWidget(self.save_results_btn)
        
        # Connect buttons to functions
        self.load_microscopy_btn.clicked.connect(self._load_microscopy)
        self.load_model_btn.clicked.connect(self._load_model)
        self.run_inference_btn.clicked.connect(self._run_inference)
        self.load_predictions_btn.clicked.connect(self._load_predictions)
        self.save_results_btn.clicked.connect(self._save_results)
        
        # Initialize variables
        self.microscopy_data = None
        self.model = None
        self.metabolites = None
        self.am_test = None
        self.annot_stats = None
        self.current_predictions = None
        self.current_metabolites = None
        
        # Disable buttons initially
        self.run_inference_btn.setEnabled(False)
        self.load_model_btn.setEnabled(False)
        inference_group.setEnabled(False)  # Disable parameters until model is loaded
        
    def _load_microscopy(self):
        """Load microscopy data and display in viewer"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Microscopy Data", "", "TIFF Files (*.tif *.tiff)"
        )
        if file_path:
            try:
                # Load microscopy data
                self.microscopy_data = tif.imread(file_path)
                print(f"Successfully loaded microscopy data from {file_path}")
                
                # Load AM test data (assuming it's in a fixed location relative to input)
                am_test_path = Path('data/am_eval.tif')
                print(f"Looking for AM test data at: {am_test_path.absolute()}")
                if am_test_path.exists():
                    self.am_test = tif.imread(str(am_test_path))[..., -1]
                    print(f"AM test data loaded from {am_test_path}")
                else:
                    print(f"Warning: AM evaluation data not found at {am_test_path}")
                    self.am_test = None  # Allow continuing without AM data
                
                # Display in viewer
                if len(self.microscopy_data.shape) == 3:  # Multi-channel image
                    for i in range(self.microscopy_data.shape[-1]):
                        self.viewer.add_image(
                            self.microscopy_data[..., i],
                            name=f'Channel_{i}',
                            colormap='viridis'
                        )
                else:  # Single channel image
                    self.viewer.add_image(
                        self.microscopy_data,
                        name='Microscopy',
                        colormap='viridis'
                    )
                
                # Enable model loading if microscopy data is loaded
                self.load_model_btn.setEnabled(True)
                print("Model loading button enabled")
                
            except Exception as e:
                print(f"Error loading microscopy data: {str(e)}")

    def _load_model(self):
        """Load the deep learning model"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Model Checkpoint", "", "Checkpoint Files (*.ckpt)"
        )
        if file_path:
            try:
                # Load annotations to get metabolites
                folder_path = 'data/training_data'
                self.metabolites, _, self.annot_stats = load_annotations(folder_path)
                
                # Load model
                self.model = load_regressor(
                    checkpoint_path=file_path,
                    num_classes=len(self.metabolites),
                    encoder='resnet152',
                    in_chans=4
                )
                self.model.eval()
                self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
                
                # Enable inference if both data and model are loaded
                self.run_inference_btn.setEnabled(True)
                self.layout().itemAt(2).widget().setEnabled(True)  # Enable inference parameters
                
            except Exception as e:
                print(f"Error loading model: {str(e)}")

    def _run_inference(self):
        """Run inference and display results"""
        if self.microscopy_data is None or self.model is None:
            print("Please load both microscopy data and model first")
            return
            
        try:
            # Calculate total steps
            eval_range = min(self.eval_range.value(), min(self.microscopy_data.shape[:2]))
            step_size = self.step_size.value()
            patches_per_dim = eval_range // step_size
            total_patches = patches_per_dim * patches_per_dim
            total_steps = total_patches // self.batch_size.value()
            if total_patches % self.batch_size.value() != 0:
                total_steps += 1  # Add one more step for remaining patches
            
            # Create progress bar
            pbar = progress(total=total_steps, desc="Running inference")
            show_info("Starting inference...")  # Show notification
            
            # Run inference with user-specified parameters
            pred_image, pred_counts = pred_images(
                test_image=self.microscopy_data,
                model=self.model,
                am_test=self.am_test,
                annot_stats=self.annot_stats,
                metabolites=self.metabolites,
                start_x=0,
                start_y=0,
                eval_range=eval_range,
                step=self.step_size.value(),
                batch_size=self.batch_size.value(),
                progress_callback=lambda: pbar.update(1)  # Update progress bar
            )
            
            # Store current predictions
            self.current_predictions = pred_image
            self.current_metabolites = self.metabolites
            
            # Display predictions
            self._display_predictions(pred_image, self.metabolites)
            
            # Add prediction counts
            self.viewer.add_image(
                pred_counts,
                name='Prediction_Counts',
                colormap='gray',
                blending='additive',
                visible=False
            )
            
            # Enable save button
            self.save_results_btn.setEnabled(True)
            
        except Exception as e:
            print(f"Error during inference: {str(e)}")

    def _load_predictions(self):
        """Load existing predicted metabolite data and display in viewer"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Predicted Metabolite Data", "", "HDF5 Files (*.h5)"
        )
        if file_path:
            try:
                with h5py.File(file_path, 'r') as f:
                    pred_data = f['pred'][:]
                    metabolites = f['metabolites'][:]
                
                # Store current predictions
                self.current_predictions = pred_data
                self.current_metabolites = metabolites
                
                # Display predictions
                self._display_predictions(pred_data, metabolites, prefix="Loaded_")
                
                # Enable save button
                self.save_results_btn.setEnabled(True)
                
            except Exception as e:
                print(f"Error loading predictions: {str(e)}")

    def _save_results(self):
        """Save current predictions to HDF5 file"""
        if self.current_predictions is None:
            print("No predictions to save")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Predictions", "", "HDF5 Files (*.h5)"
        )
        
        if file_path:
            try:
                data = {
                    'pred': self.current_predictions,
                    'metabolites': self.current_metabolites
                }
                
                with h5py.File(file_path, 'w') as f:
                    for key, value in data.items():
                        f.create_dataset(key, data=value)
                        
                print(f"Results saved to {file_path}")
                
            except Exception as e:
                print(f"Error saving results: {str(e)}")

    def _apply_gamma(self, image, gamma=2.0):
        """Apply gamma correction to image"""
        return np.power(image, 1/gamma)

    def _display_predictions(self, predictions, metabolites, prefix=""):
        """Display predictions in the viewer"""
        def scale_prediction(image, border=100):
            """Scale prediction excluding border, then apply power transformation"""
            # Create mask excluding border
            mask = np.ones_like(image, dtype=bool)
            mask[:border, :] = False
            mask[-border:, :] = False
            mask[:, :border] = False
            mask[:, -border:] = False
            
            # Get min/max excluding border and zeros
            valid_values = image[mask & (image != 0)]
            if len(valid_values) > 0:
                min_val = np.min(valid_values)
                max_val = np.max(valid_values)
                
                # Scale to [0, 1]
                scaled = np.zeros_like(image)
                scaled[mask] = (image[mask] - min_val) / (max_val - min_val)
                
                # Apply power transformation
                return 10**scaled
            return image
        
        # Display results
        for i, metabolite in enumerate(metabolites):
            name = metabolite.decode('utf-8') if isinstance(metabolite, bytes) else metabolite
            # Scale and transform each prediction independently
            pred_scaled = scale_prediction(predictions[..., i])
            
            # Apply Gaussian blur if sigma > 0
            sigma = self.sigma.value()
            if sigma > 0:
                pred_scaled = gaussian_filter(pred_scaled, sigma=sigma)
            
            self.viewer.add_image(
                pred_scaled,
                name=f'{prefix}Prediction_{name}',
                colormap='magma',
                blending='additive',
                visible=False,
                gamma=2.0
            )

@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return MetaLensWidget