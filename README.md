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

### Developer Notes

To make pyvips work on Windows:

```python
import os
vipsbin = ''
os.environ['PATH'] = vipsbin + ';' + os.environ['PATH']

import pyvips as vips
```
