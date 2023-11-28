# LazySlide

## Usage

```python
import lazyslide as zs

slide = ''  # Your SVS file
wsi = zs.WSI(slide)
wsi.plot_tissue()

wsi.create_tissue_mask()
wsi.plot_mask()

wsi.create_tiles(tile_px=256, mpp=.5)
wsi.plot_tissue(tiles=True)

# Export to pytorch dataset
wsi.to_dataset()

```

### Developer Notes

To make pyvips works on Windows:

```python
import os
vipsbin = ''
os.environ['PATH'] = vipsbin + ';' + os.environ['PATH']

import pyvips as vips
```