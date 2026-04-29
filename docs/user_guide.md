# MetaLens User Guide

MetaLens predicts super-resolved spatial metabolomics distributions from brightfield and fluorescence microscopy images using a co-trained deep learning model.

---

## 1. Installation

### Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.8 or higher |
| CUDA-capable GPU | Recommended (≥ 8 GB VRAM for inference, ≥ 16 GB for training) |
| Conda | [Miniconda](https://docs.conda.io/en/latest/miniconda.html) recommended |

### Step-by-Step Setup

```bash
# 1. Create and activate a dedicated conda environment
conda create -n metalens python=3.10
conda activate metalens

# 2. Clone the repository
git clone https://github.com/LRpz/metalens.git
cd metalens

# 3a. CLI only — no Napari GUI
pip install -e .

# 3b. Full install including Napari plugin
pip install -e .[napari]
```

---

## 2. Hardware Requirements and Performance

| Task | GPU VRAM | Expected Runtime |
|------|----------|-----------------|
| Training (16k patches, 200 epochs, batch 32) | ≥ 16 GB | ~2 hours |
| Inference (500 px region, step 2, batch 128) | ≥ 8 GB | ~3 minutes |

---

## 3. Using MetaLens on Example Data

### 3.1 Download Required Files

| File | Link |
|------|------|
| Training data | [Google Drive](https://drive.google.com/drive/folders/1ISZkGF3A9zV4Fsdke4h7qlWwZM6HuXgx?usp=sharing) |
| Ablation mark segmentation model | [Google Drive](https://drive.google.com/file/d/1l5wVWz4Xkp6-Bi1rHZLJSf5LmQQhtuKm/view?usp=sharing) |
| Pre-trained MetaLens model | [Google Drive](https://drive.google.com/file/d/1zB2kM12xB-YBJfFVMYVYYJX0sStCj2v9/view?usp=sharing) |
| Example evaluation dataset | [Google Drive](https://drive.google.com/file/d/177lS781WwD5fI_8kZPsMweXTIWiKExyj/view?usp=sharing) |

### 3.2 Place Files in the Correct Directories

```
metalens/
├── data/
│   ├── training_data/      ← Extract training patches here
│   ├── eval_dataset.tif    ← Example evaluation image
│   └── am_eval.tif         ← Ablation mark probability map for eval
└── models/
    ├── AM_segmentation.pth ← Ablation mark segmentation model
    └── pretrained_model.ckpt ← Pre-trained MetaLens model
```

### 3.3 (Optional) Re-train on Example Data

```bash
python metalens/dl/train.py data/training_data/ models/
```

### 3.4 Run Inference

#### Option A — Napari Plugin (recommended for exploration)

```bash
python -m napari
```

1. Open the plugin: **Plugins > MetaLens Viewer**
2. Load data: **Load Microscopy Data** → select `data/eval_dataset.tif`
3. Load model: **Load Model** → select `models/pretrained_model.ckpt`
4. Set inference parameters (see [Parameter Reference](#parameter-reference))
5. Click **Run Inference**
6. Click **Save Results** to export predictions as HDF5

#### Option B — Command Line

```bash
python metalens/dl/eval.py eval_dataset models/pretrained_model.ckpt
```

> **Note:** The CLI expects a *sample name*, not a full file path. It automatically reads
> `data/raw_data/<sample_name>_cells.tif` and the ablation mark map `data/am_eval.tif`.
> Place your evaluation image at `data/raw_data/eval_dataset_cells.tif` before running.

Results are saved to `data/output/output.h5`.

To visualize saved results in Napari:
```bash
python -m napari
# Plugins > MetaLens Viewer > Load Predictions → data/output/output.h5
```

---

## 4. Processing Your Own Data

### 4.1 Required Input Files

| File | Format | Description |
|------|--------|-------------|
| Pre-MALDI microscopy | Multi-channel TIFF | Stitched image with fiducials (brightfield + optional fluorescence) |
| Post-MALDI microscopy | Single-channel TIFF | Stitched brightfield showing matrix and ablation marks |
| Mass spectrometry data | `.imzML` + `.ibd` | Centroided mass spectra per ablation mark |

Place all raw files in `data/raw_data/` using the naming convention:
```
data/raw_data/
├── <dataset_name>_preMALDI_channel1.tif
├── <dataset_name>_preMALDI_channel2.tif   (optional)
├── <dataset_name>_postMALDI_channel1.tif
├── <dataset_name>.imzML
└── <dataset_name>.ibd
```

---

### 4.2 Preprocessing Pipeline

Run each step in order, replacing `sample_001` with your dataset name.

#### Step 1 — Register and Crop Microscopy Images

Aligns pre- and post-MALDI images and crops to the region of interest.

```bash
python metalens/preprocessing/microscopy_registration_crop.py sample_001
```

| | Path |
|--|------|
| **Input** | `data/raw_data/sample_001_preMALDI_channel*.tif` |
| | `data/raw_data/sample_001_postMALDI_channel1.tif` |
| **Output** | `data/raw_data/sample_001_cells.tif` |
| | `data/raw_data/sample_001_ablation_marks_tf.tif` |

---

#### Step 2 — Cell Segmentation

Segments cells using Cellpose's `cyto2` generalist model.

```bash
python metalens/preprocessing/cell_segmentation.py sample_001
```

| | Path |
|--|------|
| **Input** | `data/raw_data/sample_001_cells.tif` |
| **Output** | `data/raw_data/sample_001_cells_mask.tif` |

---

#### Step 3 — Ablation Mark Segmentation

Segments ablation marks using a custom pre-trained U-Net, optimized for 10× brightfield
microscopy with DAN or DHB matrix.

```bash
python metalens/preprocessing/AM_segmenation_inference.py sample_001
```

| | Path |
|--|------|
| **Input** | `data/raw_data/sample_001_ablation_marks_tf.tif` |
| **Output** | `data/raw_data/sample_001_ablation_marks_tf_pred.tif` |

---

#### Step 4 — Generate Training Patches

Extracts image patches co-registered with mass spectrometry intensities.

```bash
python metalens/preprocessing/make_training_patches.py sample_001
```

| | Path |
|--|------|
| **Input** | All outputs from Steps 1–3 |
| | `data/raw_data/sample_001.imzML` + `.ibd` |
| **Output** | `data/training_data/sample_001_*.tif` (one file per patch) |
| | `data/training_data/ion_intensities.csv` |

---

### 4.3 Training

```bash
python metalens/dl/train.py <training_data_folder> <model_output_folder>
```

**Example:**
```bash
python metalens/dl/train.py data/training_data models/
```

**Default hyperparameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 32 | Training batch size |
| `learning_rate` | 1e-3 | Initial learning rate (halved every 50 epochs) |
| `epochs` | 200 | Total training epochs |
| `encoder` | resnet152 | Backbone architecture |

**Expected inputs:**

- `<training_data_folder>/ion_intensities.csv` — metabolite intensities per patch
- `<training_data_folder>/<sample>_<id>.tif` — training patch images

**Output:** Model checkpoints saved to `<model_output_folder>/` as:
```
{epoch}-{val_loss}-{val_pearson_corrcoef}-{val_r2_score}.ckpt
```
TensorBoard logs are also written for monitoring training progress (`tensorboard --logdir models/`).

---

### 4.4 Inference

#### Via Napari Plugin

```bash
python -m napari
```

1. Open: **Plugins > MetaLens Viewer**
2. **Load Microscopy Data** → select your cells `.tif`
3. **Load Model** → select your `.ckpt` checkpoint
4. Configure parameters (see [Parameter Reference](#parameter-reference))
5. **Run Inference**
6. **Save Results** to export as HDF5

#### Via Command Line

```bash
python metalens/dl/eval.py <sample_name> <model_path>
```

**Example:**
```bash
python metalens/dl/eval.py sample_001 models/best_model.ckpt
```

The script reads `data/raw_data/<sample_name>_cells.tif` and writes predictions to
`data/output/output.h5`.

**HDF5 output structure:**

| Key | Shape | Description |
|-----|-------|-------------|
| `pred` | `(H, W, N)` | Predicted metabolite intensity maps |
| `metabolites` | `(N,)` | Metabolite name for each channel |

---

## 5. Parameter Reference

### Inference Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| **Evaluation Range** | 500 px | Side length (in pixels) of the square region to analyze. Smaller values are faster. |
| **Step Size** | 2 px | Pixel stride between consecutive predictions. Lower = more accurate but slower. Combine `step_size=2` with Gaussian blur σ=2 for fast, smooth maps. |
| **Batch Size** | 128 | Patches processed per forward pass. Reduce if you run out of GPU memory. |

### Display Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| **Gaussian Blur σ** | 0 | Smoothing applied before display. Useful when using a large step size. |

---

## 6. Complete Pipeline Example

```bash
conda activate metalens

# Place raw data in data/raw_data/ as sample_001_*

# Preprocessing
python metalens/preprocessing/microscopy_registration_crop.py sample_001
python metalens/preprocessing/cell_segmentation.py sample_001
python metalens/preprocessing/AM_segmenation_inference.py sample_001
python metalens/preprocessing/make_training_patches.py sample_001

# Training
python metalens/dl/train.py data/training_data models/

# Inference
python metalens/dl/eval.py sample_001 models/<checkpoint>.ckpt
```

---

## 7. Expected Directory Structure

After successfully running MetaLens on a dataset named `sample_001`:

```
metalens/
├── data/
│   ├── raw_data/
│   │   ├── sample_001_preMALDI_channel1.tif
│   │   ├── sample_001_postMALDI_channel1.tif
│   │   ├── sample_001_cells.tif
│   │   ├── sample_001_cells_mask.tif
│   │   ├── sample_001_ablation_marks_tf.tif
│   │   ├── sample_001_ablation_marks_tf_pred.tif
│   │   ├── sample_001.imzML
│   │   └── sample_001.ibd
│   ├── training_data/
│   │   ├── sample_001_0.tif
│   │   ├── sample_001_1.tif
│   │   ├── ...
│   │   └── ion_intensities.csv
│   └── output/
│       └── output.h5
├── models/
│   ├── AM_segmentation.pth
│   └── <epoch>-<val_loss>-<val_pearson>-<val_r2>.ckpt
└── metalens/
    ├── dl/
    ├── napari/
    └── preprocessing/
```
