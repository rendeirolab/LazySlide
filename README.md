# LazySlide


## Installation

```bash
pip install git+ssh://git@github.com/rendeirolab/LazySlide.git
conda install -c conda-forge pyvips
```

## Usage

```python
import lazyslide as zs

slide = 'https://github.com/camicroscope/Distro/raw/master/images/sample.svs'  # Your SVS file
wsi = zs.WSI(slide)
wsi.plot_tissue()

wsi.create_tissue_mask()
wsi.plot_mask()

wsi.create_tiles(tile_px=256, mpp=.5)
wsi.plot_tissue(tiles=True)

# Alternatively, you can create tiles by contours
wsi.create_tissue_contours()
wsi.plot_tissue(contours=True)

wsi.create_tiles(tile_px=256, mpp=.5)
wsi.plot_tissue(tiles=True)
```

To do feature extraction:
```python
import torch
from torch.utils.data import DataLoader
from lazyslide.loader import FeatureExtractionDataset

loader = DataLoader(
    dataset=FeatureExtractionDataset(wsi, resize=224, color_normalize="macenko"), 
    batch_size=16)

resnet = torch.hub.load("pytorch/vision", "resnet50", weights="IMAGENET1K_V2")

with torch.no_grad():
    for tile in loader:
        tile_feature = resnet(tile)
```

To set lazyslide to use different slide readers:

```python
import lazyslide as zs

slide = 'https://github.com/camicroscope/Distro/raw/master/images/sample.svs'  # Your SVS file
wsi = zs.WSI(slide, reader='openslide')
wsi = zs.WSI(slide, reader='vips')
wsi = zs.WSI(slide, reader='cucim')

```

By default, lazyslide will select an available one for you.

### Developer Notes

We use pre-commit hooks to ensure code quality.

```bash
# Install pre-commit
pip install pre-commit

# Install pre-commit hooks
pre-commit install

# Run pre-commit hooks on all files
pre-commit run --all-files
```

To make pyvips work on Windows:

```python
import os
vipsbin = ''
os.environ['PATH'] = vipsbin + ';' + os.environ['PATH']

import pyvips as vips
```

```shell

nextflow run lazyslide --annotations slides.csv --output output --tiles 256 \
--mpp 0.5 --model resnet50 --batch 16 --color_normalize macenko

```
