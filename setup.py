from setuptools import setup

setup(
    name='metalens',
    version='0.1.0',
    description='MetaLens: Super-Resolved Spatial Metabolomics',
    packages=['MetaLens'],
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
            'metalens = MetaLens.napari.plugin',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)