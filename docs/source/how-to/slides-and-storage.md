# Slides and storage

## How do I open my own slide?

```python
import lazyslide as zs

wsi = zs.open_wsi("path/to/slide.svs")
```

If the format needs a non-default reader, install it first and let `wsidata` detect it. See [supported readers and formats](../reference/formats).

## How do I inspect dimensions, MPP, magnification, and levels?

Inspect the slide in Python:

```python
print(wsi.properties)
print(wsi.fetch.pyramids())
```

Treat missing, zero, or implausible MPP as a data-quality problem before running models with a physical-resolution requirement.

## How do I save and reopen an analysis?

Choose the backing store when opening the source slide:

```python
wsi = zs.open_wsi("slide.svs", backed_file="slide-analysis.zarr")
# Run analysis steps.
wsi.write(overwrite=True)
```

Reopen with the same source slide and backing store:

```python
wsi = zs.open_wsi("slide.svs", backed_file="slide-analysis.zarr")
```

The WSI pixels remain in the source slide; generated shapes, tables, images, and metadata live in the backing store.

## How do I inspect what has already been computed?

```python
print("shapes:", list(wsi.shapes))
print("tables:", list(wsi.tables))
print("images:", list(wsi.images))
print("metadata:", list(wsi.attrs))
```

Also inspect the relevant shape or table directly:

```python
print(wsi.shapes["tiles"].head())
print(wsi.tables["uni_tiles"])
```

## How do I keep several analyses in one object?

Give each result a descriptive key:

```python
zs.pp.find_tissues(wsi, key_added="tissues_otsu")
zs.pp.find_tissues(wsi, method="entropy", key_added="tissues_entropy")

zs.pp.tile_tissues(
    wsi,
    256,
    mpp=0.5,
    tissue_key="tissues_entropy",
    key_added="tiles_20x",
)
```

Use the same explicit key in downstream calls. Avoid silently replacing a tile table after computing features associated with it.
