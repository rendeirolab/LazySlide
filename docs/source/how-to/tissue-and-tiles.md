# Tissue detection and tiling

## How do I choose a tissue detection method?

Start with the fast default method:

```python
zs.pp.find_tissues(wsi, method="otsu")
```

For faint or texturally distinct tissue, compare entropy-based detection:

```python
zs.pp.find_tissues(wsi, method="entropy", key_added="tissues_entropy")
```

For learned tissue segmentation, use:

```python
zs.seg.tissue(wsi, model="grandqc", key_added="tissues_dl")
```

Always visualize the result and choose based on the tissue and stain being analyzed.

## How do I fix missing or poor tissue detection?

Try these checks in order:

1. Confirm that the slide opens correctly and is not inverted or multi-channel data interpreted as RGB.
2. Run at a fixed level so results are reproducible: `level=-1` is the lowest-resolution level.
3. Compare `method="otsu"` and `method="entropy"`.
4. For Otsu, adjust `threshold`, `filter_artifacts`, or `to_hsv`.
5. Reduce `min_tissue_area` when legitimate fragments are removed.
6. Use `refine_level="auto"` when coarse boundaries need refinement.
7. Compare a learned tissue model.

```python
zs.pp.find_tissues(
    wsi,
    level=-1,
    method="entropy",
    min_tissue_area=1e-4,
    key_added="tissues_relaxed",
)
zs.pl.tissue(wsi, tissue_key="tissues_relaxed")
```

## How do I control holes and small fragments?

```python
zs.pp.find_tissues(
    wsi,
    detect_holes=True,
    min_tissue_area=1e-3,
    min_hole_area=1e-5,
)
```

Areas are proportions of the slide area. Increase a threshold to remove smaller objects; decrease it to preserve them.

## How do I choose tile size and MPP?

Follow the model's documented input first. Otherwise decide the biological field of view you need:

```text
field of view (µm) = tile_px * mpp
```

For example, `tile_px=256, mpp=0.5` covers 128 µm. See [Resolution and tiling](../concepts/resolution) for a comparison table.

## How do I create overlapping tiles?

Use either an overlap ratio or an explicit stride:

```python
zs.pp.tile_tissues(wsi, 256, mpp=0.5, overlap=0.25)
# or
zs.pp.tile_tissues(wsi, 256, mpp=0.5, stride_px=192)
```

Do not specify both `overlap` and `stride_px`.

## How do I make several tile sets?

```python
zs.pp.tile_tissues(wsi, 256, mpp=0.5, key_added="tiles_20x")
zs.pp.tile_tissues(wsi, 256, mpp=1.0, key_added="tiles_10x")
```

Pass the selected key to every downstream operation:

```python
zs.tl.feature_extraction(wsi, "uni", tile_key="tiles_20x")
```

## Why are no tiles generated?

Common causes are:

- the tile is larger than every detected tissue region;
- tissue detection produced no contours;
- `background_fraction` rejects border tiles;
- the requested MPP and slide metadata imply an unexpectedly large level-0 footprint.

Inspect tissue polygons, lower `tile_px`, relax `background_fraction`, or temporarily set `background_filter=False` to isolate the cause.

```python
zs.pp.tile_tissues(
    wsi,
    256,
    mpp=0.5,
    background_filter=True,
    background_fraction=0.5,
)
```

## How do I tile the whole slide or an annotation?

To tile the full slide, set `tissue_key=None`. To tile an imported annotation layer, use its shape key:

```python
zs.pp.tile_tissues(wsi, 256, mpp=0.5, tissue_key=None, key_added="all_tiles")
zs.pp.tile_tissues(
    wsi, 256, mpp=0.5, tissue_key="annotations", key_added="roi_tiles"
)
```
