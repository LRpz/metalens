# MetaLens User Guide

MetaLens is a tool for generating super-resolved spatial metabolomics data from microscopy images. This documentation provides detailed instructions for installation and usage.

## 1. Installation

### 1.1. Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU with 16GB+ RAM
- [Conda](https://docs.conda.io/en/latest/miniconda.html) (recommended)

### 1.2. Step-by-Step Installation
```bash
# 1. Create and activate conda environment
conda create -n metalens python=3.10
conda activate metalens

# 2. Install MetaLens
git clone https://github.com/LRpz/metalens.git
cd ./metalens

# For command-line interface only:
pip install -e .

# For full installation with Napari plugin:
pip install -e .[napari]
```

## 2. Using MetaLens

Hardware Requirements and Performance:

- GPU: CUDA-capable GPU with 16GB+ VRAM required

- Training Performance:
    - Dataset size: 16,000 patches
    - Training parameters:
        - `epochs`: 200
        - `batch_size`: 32
    - Expected runtime: ~2 hours

- Inference Performance:
    - Example configuration:
        - `evaluation_range`: 500 pixels (size of region to analyze)
        - `batch_size`: 128 patches
        - `step_size`: 2 pixels (distance between predictions)
    - Expected runtime: ~3 minutes

### 2.1. On Example Data

1. Download the required files:

    - Training data from [this link](https://drive.google.com/drive/folders/1ISZkGF3A9zV4Fsdke4h7qlWwZM6HuXgx?usp=sharing)
    - Ablation Mark segmentation model from [this link](https://drive.google.com/file/d/1l5wVWz4Xkp6-Bi1rHZLJSf5LmQQhtuKm/view?usp=sharing)
    - Trained MetaLens model from [this link](https://drive.google.com/file/d/1zB2kM12xB-YBJfFVMYVYYJX0sStCj2v9/view?usp=sharing)
    - Example evaluation dataset from [this link](https://drive.google.com/file/d/177lS781WwD5fI_8kZPsMweXTIWiKExyj/view?usp=sharing)

2. Place the files in their respective directories:
    - Extract training patches to `data/training_data/`
    - Place `AM_segmentation.pth` in `models/`
    - Place `trained_model.ckpt` in `models/`
    - Place `eval_dataset.tif` in `data/`

3. (Optional) Re-train on exemple training data: 
    ```bash
    python metalens/dl/train.py data/training_data/ models/
    ```

3. Run inference on the test dataset:

    **Using napari**

    ```bash
    python -m napari
    ```

    - In napari's GUI `Plugins > Metalens`

    - Load test dataset `Load Microsocpy Data > data/eval_dataset.tif`
    - Load trained model `Load Model > models/trained_model.ckpt` (*Adjust model path if using a newly trained model*)
    - Adjust inference parameters:
        - Evaluation Range: Size of region to analyze, in pixels
        - Step Size: Distance between predictions in pixels\
          *Note: Lower step size provides more accurate predictions but increases computation time. 
          For visualization purposes, you can use larger step sizes combined with Gaussian blur 
          (recommended: step_size=2 with Gaussian blur sigma=2)*
        - Batch Size: Number of patches per batch, adjust based on available VRAM
   - Run inference using `Run Inference`
   - Save results using `Save Results`
   - Adjust vizualisation options using Napari's controls

    **Using command line**

    - Run
        ```bash 
        python metalens/dl/eval.py data/eval_dataset.tif models/trained_model.ckpt
        ```

    - Results data will be saved as `data/output.h5`

    - Visualize using Napari:
        ```bash
        python -m napari
        ```

    - In napari's GUI `Plugins > Metalens > Load Predictions > data/output/output.h5`



### 2.2. On New Data
#### 2.2.1. Required Data Format
Your input data should include:

1. **Pre-MALDI Microscopy Images**

    - Format: Multi-channel TIFF (.tif)
    - Channels: 
        - Brightfield
        - Fluorescence (optional)
    - Content: Stitched tiled microscopy with fiducials showing the biological sample
    
2. **Post-MALDI Microscopy Images**
    - Format: Single-channel TIFF (.tif)
    - Channels: 
        - Brightfield
    - Content: Stitched tiled microscopy with fiducials showing the matrix and ablation marks

3. **Mass Spectrometry Data**
    - Format: .imzML and .ibd
    - Content: Centroided mass spectra for each ablation mark

#### 2.2.2. Preprocessing Pipeline

#### 1. Register and Crop Microscopy Images
This step aligns pre- and post-MALDI images and crops them to the region of interest.

```bash
python metalens/preprocessing/microscopy_registration_crop.py <dataset_name>
```

Expected input files:

- `data/raw_data/<dataset_name>_preMALDI_channel1.tif`
- `data/raw_data/<dataset_name>_postMALDI_channel1.tif`

Output files:

- `data/raw_data/<dataset_name>_cells.tif`
- `data/raw_data/<dataset_name>_ablation_marks_tf.tif`

#### 2. Cell Segmentation
This step performs cell segmentation using Cellpose's generalist model 'cyto2'.

```bash
python metalens/preprocessing/cell_segmentation.py <dataset_name>
```

Expected input:

- `data/raw_data/<dataset_name>_cells.tif`

Output:

- `data/raw_data/<dataset_name>_cells_mask.tif`

#### 3. Ablation Mark Segmentation
This step segments ablation marks using a custom pre-trained model. This model has been optimized to segment brightfield microscopy at 10X magnification of non overlapping ablation marks with DAN and DHB matrices.

   ```bash
   python metalens/preprocessing/AM_segmenation_inference.py <dataset_name>
   ```

Expected input:

- `data/raw_data/<dataset_name>_ablation_marks_tf.tif`

Output:

- `data/raw_data/<dataset_name>_ablation_marks_tf_pred.tif`

#### 4. Generate Training Patches
This step creates training patches by combining microscopy and mass spec data.

```bash
python metalens/preprocessing/make_training_patches.py <dataset_name>
```

Expected inputs:

- All previous outputs
- Mass spec data:
    - `data/raw_data/<dataset_name>.imzML`
    - `data/raw_data/<dataset_name>.ibd`

Output:

- Training patches in `data/training_data/`
- `data/training_data/ion_intensities.csv`

### 2.2.3. Training 

   ```bash
   python metalens/dl/train.py <training_data_folder> <model_output_folder>
   ```
   
   Parameters:

   - `batch_size`: Default 32
   - `learning_rate`: Default 1e-3
   - `epochs`: Default 200

   Expected inputs:

   - Training data folder containing:

       - `ion_intensities.csv`: Metabolite intensities for each patch
       - Training patches (.tif files) referenced in the CSV
   
   Output:

   - Model checkpoints in `<model_output_folder>`:

       - Best model: `{epoch}-{val_loss}-{val_pearson_corrcoef}-{val_r2_score}.ckpt`
       - TensorBoard logs for training monitoring

### 2.2.4. Inference

#### 2.2.4.1. Via Napari Plugin

1. Launch Napari:
   ```bash
   python -m napari
   ```

2. Load MetaLens plugin:

    - Navigate to `Plugins > MetaLens`
    - Click "MetaLens Viewer"

3. Use the plugin:
    - Load microscopy data using "Load Microscopy Data"
    - Load model using "Load Model"
    - Adjust inference parameters:

        - Evaluation Range: Size of region to analyze, in pixels
        - Step Size: Distance between predictions in pixels\
        *Note: Lower step size provides more accurate predictions but increases computation time. 
        For visualization purposes, you can use larger step sizes combined with Gaussian blur 
        (recommended: step_size=2 with Gaussian blur sigma=2)*
        - Batch Size: Number of patches per batch

    - Run inference using "Run Inference"
    - Save results using "Save Results"

#### 2.2.4.2. Via Command Line

   ```bash
   python metalens/dl/eval.py <input_image> <model_path>
   ```

   Expected inputs:

   - `<input_image>`: Multi-channel TIFF file of the cells
   - `<model_path>`: Path to trained model checkpoint (.ckpt file)
   
   Output:

   - HDF5 file containing:

       - `pred`: Predicted metabolite intensities (height × width × n_metabolites)
       - `metabolites`: List of predicted metabolite names

### 2.3. Complete Pipeline Example
The following example shows how to process your own data from scratch:

1. Prepare environment:
    ```bash
    conda activate metalens
    ```

2. Prepare your data:
    Place your microscopy and mass spec files in `data/raw_data/`

3. Preprocess data:
    ```bash
    python metalens/preprocessing/microscopy_registration_crop.py sample_001
    python metalens/preprocessing/cell_segmentation.py sample_001
    python metalens/preprocessing/AM_segmenation_inference.py sample_001
    python metalens/preprocessing/make_training_patches.py sample_001
    ```

4. Train model
    ```bash
    python metalens/dl/train.py data/training_data models
    ```

5. Run inference
    ```bash
    python metalens/dl/eval.py data/eval_dataset.tif models/trained_model.ckpt
    ```

### 2.4. Expected Directory Structure
After sucessfully running MetaLens on a new dataset named `sample_001`, the directory structure should be:
```bash
metalens/
├── data/
│   ├── raw_data/
│   │   ├── sample_001_preMALDI_channel1.tif
│   │   ├── sample_001_postMALDI_channel1.tif
│   │   ├── sample_001_cells.tif
│   │   ├── sample_001_ablation_marks_tf.tif`
│   │   ├── sample_001.imzML
│   │   ├── sample_001.ibd
│   │   ├── sample_001_ablation_marks_tf_pred.tif
│   │   └── sample_001_cells_mask.tif 
│   ├── training_data/
│   │   ├── sample_001_0.tif
│   │   ├── sample_001_0.tif
│   │   ├── sample_001_0.tif
│   │   ├── ...
│   │   └── ion_intensities.csv
│   └── output/
│       └──output.h5
├── models/
│   ├── AM_segmentation.pth
│   └── trained_model.ckpt
└── metalens/
    ├── dl/
    │   ├── __init__.py
    │   ├── eval.py
    │   ├── train.py
    │   └── utils.py
    ├── napari/
    │   ├── __init__.py
    │   ├── plugin.py
    │   └── napari.yaml
    └── preprocessing/
        ├── AM_segmenation_inference.py
        ├── cell_segmentation.py
        ├── make_training_patches.py
        └── microscopy_registration_crop.py
``` 