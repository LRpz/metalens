from setuptools import setup, find_packages

setup(
    name='metalens',
    version='0.1.0',
    description='MetaLens: Super-Resolved Spatial Metabolomics',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'numpy',
        'torch',
        'tifffile',
        'h5py',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'albumentations',
        'pytorch-lightning',
        'segmentation-models-pytorch'
    ],
    extras_require={
        'napari': [
            'napari[all]>=0.4.19',
            'magicgui>=0.8.2',
            'qtpy',
            'napari-plugin-engine',
        ],
    },
    entry_points={
        'napari.plugin': [
            'metalens = metalens.napari.plugin',
        ],
    },
)