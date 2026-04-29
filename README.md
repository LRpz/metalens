# MetaLens

<p align="center">
  <strong>Super-resolved spatial metabolomics from microscopy images using deep learning.</strong>
</p>

<p align="center">
  <a href="https://www.biorxiv.org/content/10.1101/2024.08.29.610242v1">
    <img src="https://img.shields.io/badge/paper-biorXiv-b31b1b?style=flat-square" alt="Paper"/>
  </a>
  <a href="https://lrpz.github.io/metalens/">
    <img src="https://img.shields.io/badge/docs-mkdocs-4051b5?style=flat-square" alt="Docs"/>
  </a>
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/python-3.8%2B-3776ab?style=flat-square&logo=python&logoColor=white" alt="Python 3.8+"/>
  </a>
  <a href="http://creativecommons.org/licenses/by-nc/4.0/">
    <img src="https://img.shields.io/badge/license-CC%20BY--NC%204.0-lightgrey?style=flat-square" alt="License: CC BY-NC 4.0"/>
  </a>
</p>

<p align="center">
  <img src="docs/assets/napari-demo.gif" alt="MetaLens Napari Plugin Demo" width="800"/>
</p>

---

## What is MetaLens?

MetaLens is a deep learning tool that predicts spatial metabolomics distributions at single-cell resolution directly from brightfield and fluorescence microscopy images. It bridges the gap between high-resolution microscopy and low-resolution MALDI mass spectrometry imaging, enabling super-resolved metabolite maps without additional experimental cost.

The approach trains a convolutional model (DeepLabV3+ backbone) on co-registered microscopy and MALDI-MSI data, then applies it pixel-by-pixel across new tissue sections.

**Associated publication:** [Inferring super-resolved spatial metabolomics from microscopy](https://www.biorxiv.org/content/10.1101/2024.08.29.610242v1)

## Key Features

- **No-code inference** via an interactive [Napari](https://napari.org) plugin
- **Full preprocessing pipeline** for co-registering pre/post-MALDI microscopy images with mass spectrometry data
- **Automated segmentation** of cells (Cellpose) and ablation marks (custom U-Net)
- **Flexible training** on your own datasets with configurable hyperparameters
- **HDF5 output** for convenient downstream analysis

## Quick Start

```bash
# 1. Create environment
conda create -n metalens python=3.10
conda activate metalens

# 2. Install (with Napari plugin)
git clone https://github.com/LRpz/metalens.git
cd metalens
pip install -e .[napari]

# 3. Launch Napari plugin
python -m napari
# Then: Plugins > MetaLens Viewer
```

For full installation instructions, dataset downloads, and pipeline usage see the **[User Guide](https://lrpz.github.io/metalens/user_guide/)**.

## Repository Structure

```
metalens/
├── metalens/
│   ├── dl/                 # Deep learning (training, inference, model)
│   ├── napari/             # Napari plugin
│   └── preprocessing/      # Image registration, segmentation, patch extraction
├── models/                 # Pre-trained model weights
├── data/                   # Example datasets
└── docs/                   # Documentation source
```

## Hardware Requirements

| Task | Requirement |
|------|-------------|
| Inference | CUDA GPU, ≥ 8 GB VRAM |
| Training | CUDA GPU, ≥ 16 GB VRAM |
| CPU-only | Supported for inference (slower) |

## Citation

If you use MetaLens in your research, please cite:

```bibtex
@article{rappez2024metalens,
  title   = {Inferring super-resolved spatial metabolomics from microscopy},
  author  = {Rappez, Luca and others},
  journal = {bioRxiv},
  year    = {2024},
  doi     = {10.1101/2024.08.29.610242}
}
```

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License](http://creativecommons.org/licenses/by-nc/4.0/). See the `LICENSE` file for details.
