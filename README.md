# LazySlide

LazySlide is a Python package for whole-slide image (WSI) processing. 
It is designed to be fast and memory-efficient, allowing users to work 
with large WSIs on modest hardware.

- Wide supported format
- Remote file support
- `scanpy`-style API
- CLI and Nextflow support


## Installation

PYPI

```bash
pip install lazyslide
```

Conda/Mamba (Not available yet)

```bash
conda install -c conda-forge lazyslide
#or
mamba install -c conda-forge lazyslide
```

## CLI Usage

You can launch lazyslide as a command-line tool. 

To get slide information

```shell
lazyslide info your.svs
```

To run a feature extraction pipeline

```shell
# To preprocess the slide
lazyslide preprocess your.svs --tile-size 256 --mpp 0.5
# Generate a qc report for the preprocess step
lazyslide report your.svs
# Run the feature extraction pipeline
lazyslide features your.svs --model resnet50 --batch 16 --color-normalize macenko
```
