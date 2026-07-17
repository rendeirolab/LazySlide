# Analyze your first slide

This example uses a bundled sample, requires no external slide file, and demonstrates the core LazySlide workflow.

## 1. Load and inspect a slide

```python
import lazyslide as zs

wsi = zs.datasets.sample(with_data=False)
print(wsi.properties)
```

For your own file, replace the second line with:

```python
wsi = zs.open_wsi("path/to/slide.svs")
```

Check that the dimensions, MPP, and magnification are plausible before running a physical-resolution workflow.

## 2. Find tissue

```python
zs.pp.find_tissues(wsi)
zs.pl.tissue(wsi)
```

This adds tissue polygons under `wsi.shapes["tissues"]`. If the stain is faint or unusual, see [Tissue detection and tiling](../how-to/tissue-and-tiles).

## 3. Create tiles

```python
zs.pp.tile_tissues(wsi, tile_px=256, mpp=0.5)
zs.pl.tiles(wsi, linewidth=0.5)
```

This records tile coordinates under `wsi.shapes["tiles"]`; it does not duplicate the complete WSI in memory.

## 4. Extract features

```python
zs.tl.feature_extraction(wsi, model="resnet50")
```

The default output key combines the model and tile-set names, for example `resnet50_tiles`. Inspect available content with:

```python
print(wsi.shapes.keys())
print(wsi.tables.keys())
```

## 5. Visualize a feature

```python
zs.pl.tiles(wsi, feature_key="resnet50", color=["1", "99"])
```

## 6. Save your work

For persistent analysis, open the slide with a backing store and then write it:

```python
wsi = zs.open_wsi("path/to/slide.svs", backed_file="analysis.zarr")
# Run the processing steps above.
wsi.write(overwrite=True)
```

The original slide remains separate; the backing store contains the analysis data. Continue with [the data model](../concepts/data-model), [preprocessing tutorial](../tutorials/preprocessing), or [How-To guides](../how-to/index).
