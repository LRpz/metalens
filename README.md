# MetaLens: Codebase for Super-Resolved Spatial Metabolomics

This repository contains the codebase for the paper titled [**"Inferring super-resolved spatial metabolomics from microscopy"**](https://www.biorxiv.org/content/10.1101/2024.08.29.610242v1).

## Table of Contents
1. [Introduction](#introduction)
2. [Quick Start](#quick-start)
3. [Documentation](#documentation)
4. [License](#license)
5. [Acknowledgements](#acknowledgements)

## Introduction

MetaLens is a tool for generating super-resolved spatial metabolomics data from microscopy images. The codebase and accompanying datasets allow users to replicate and extend the experiments presented in the associated research paper.

## Quick Start

MetaLens provides a user-friendly Napari plugin for no-code inference:

<p align="center">
  <img src="docs/assets/napari-demo.gif" alt="MetaLens Napari Plugin Demo" width="800"/>
</p>

1. Download the example data and models from our [Google Drive](https://drive.google.com/drive/folders/1ISZkGF3A9zV4Fsdke4h7qlWwZM6HuXgx?usp=sharing)
2. Install MetaLens with Napari support: `pip install -e .[napari]`
3. Launch Napari and access MetaLens through `Plugins > MetaLens`

## Documentation

For detailed installation and usage instructions, please refer to our [User Guide](docs/user_guide.md).

## License

This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License. To view a copy of this license, visit http://creativecommons.org/licenses/by-nc/4.0/ . See the `LICENSE` file for details.

## Acknowledgements

We acknowledge the contributors to the SpaceM dataset and the developers of the tools integrated into this pipeline. Special thanks to the research community that has supported this work.
