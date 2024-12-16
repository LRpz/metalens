# MetaLens: Codebase for Super-Resolved Spatial Metabolomics

This repository contains the codebase for the paper titled [**"Inferring super-resolved spatial metabolomics from microscopy"**](https://www.biorxiv.org/content/10.1101/2024.08.29.610242v1).

## Table of Contents
1. [Introduction](#introduction)
2. [Setup](#setup)
3. [Napari Integration](#napari-integration)
4. [Training Data Preparation](#training-data-preparation)
5. [Model Training](#model-training)
6. [Model Inference](#model-inference)
7. [License](#license)
8. [Acknowledgements](#acknowledgements)

## Introduction

MetaLens is a tool for generating super-resolved spatial metabolomics data from microscopy images. The codebase and accompanying datasets allow users to replicate and extend the experiments presented in the associated research paper.

## Setup

To set up the Python environment required for MetaLens, follow these steps:

```bash
conda create -n MetaLens python==3.10
conda activate MetaLens

# Basic installation (CLI only)
pip install -e .

# Full installation (including Napari plugin)
pip install -e .[napari]
```

**Note:** A CUDA-capable machine with a GPU having more than 16GB of RAM is strongly recommended for both training and evaluation tasks. The expected runtime for training is approximately 2 hours, while evaluation can take up to 12 hours.

## Napari Integration

MetaLens provides a Napari plugin for interactive visualization and analysis. The plugin offers a user-friendly interface for model inference and results visualization. After installing MetaLens with Napari support, launch Napari and access the plugin through the `Plugins > MetaLens` menu.

<p align="center">
  <img src="docs/assets/napari-demo.gif" alt="MetaLens Napari Plugin Demo" width="800"/>
</p>

## Training Data Preparation

The training data necessary for model development can be obtained and prepared as follows:

### Download Pre-prepared Training Data

1. Access the training data from [this link](https://drive.google.com/drive/folders/1ISZkGF3A9zV4Fsdke4h7qlWwZM6HuXgx?usp=sharing).
2. Download and extract the patches to the `data/training_data` directory.

### Generate New Training Data

To generate new training data using other SpaceM datasets, follow these steps:

1. Download the dataset archive from [this link](https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BSST369).
2. In the archive, navigate to `IDR_submission_Rappez > Data > ExperimentB` and copy all files that start with the same prefix (e.g., `F1`, `F2`, `F3`, `F4`) into the `data/raw_data` directory.

For the dataset `F1`, repeat the following steps for other datasets:

#### a. Register and Crop Pre- and Post-MALDI Data

```bash
python metalens/preprocessing/microscopy_registration_crop.py F1
```

#### b. Segment Cells

```bash
python metalens/preprocessing/cell_segmentation.py F1
```

#### c. Segment Ablation Marks Using Pre-Trained Model

1. Download the Ablation Mark (AM) segmentation model from [this link](https://drive.google.com/file/d/1l5wVWz4Xkp6-Bi1rHZLJSf5LmQQhtuKm/view?usp=drive_link).
2. Place the downloaded model in the `models` directory.
3. Run the segmentation:

```bash
python metalens/preprocessing/AM_segmenation_inference.py F1
```

#### d. Generate Training Patches

```bash
python metalens/preprocessing/make_training_patches.py F1
```

## Model Training

You can either use the pretrained model or train a new model:

### Using Pretrained Model

1. Download the pretrained model from [this link](https://drive.google.com/file/d/1zB2kM12xB-YBJfFVMYVYYJX0sStCj2v9/view?usp=drive_link).
2. Place the model in the `models` directory.

### Train a New Model

```bash
python metalens/dl/train.py data/training_data models
```

## Model Inference

To evaluate a model on a dataset (e.g., `eval_dataset.tif`, download [here](https://drive.google.com/file/d/177lS781WwD5fI_8kZPsMweXTIWiKExyj/view?usp=drive_link) and place in the `data` directory), use the following command:

```bash
python metalens/dl/eval.py data/eval_dataset.tif models/pretrained_model.ckpt
```

## License

This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License. To view a copy of this license, visit http://creativecommons.org/licenses/by-nc/4.0/ . See the `LICENSE` file for details.

## Acknowledgements

We acknowledge the contributors to the SpaceM dataset and the developers of the tools integrated into this pipeline. Special thanks to the research community that has supported this work.
