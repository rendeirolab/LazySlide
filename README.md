# LazySlide

LazySlide is a Python package for whole-slide image (WSI) processing. 
It is designed to be fast and memory-efficient, allowing users to work 
with large WSIs on modest hardware.

- Wide supported format
- Remote file support
- `scanpy`-style API
- `SpatialData` backed
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
lazyslide tissue your.svs
lazyslide tile your.svs --tile-size 256 --mpp 0.5
lazyslide features your.svs --model resnet50 --batch 16 --color-normalize macenko
lazyslide anndata your.svs --features resnet50 --output output.h5ad
```

or run it at once

```shell
lazyslide feature_pipeline your.svs --tile-size 256 --mpp 0.5 \
      --model resnet50 --batch 16 --color-normalize macenko --output output.h5ad
```


## Nextflow Usage

```shell
nextflow run lazyslide --annotations slides.csv --output output --tiles 256 \
--mpp 0.5 --model resnet50 --batch 16 --color_normalize macenko
```
