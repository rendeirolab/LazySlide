# Segmentation

## How do I choose between traditional and learned tissue segmentation?

Use `zs.pp.find_tissues` for a fast, interpretable baseline. Use `zs.seg.tissue` when color or texture thresholds are unreliable or when a model matches the slide domain.

```python
zs.pp.find_tissues(wsi, method="otsu", key_added="tissues_otsu")
zs.seg.tissue(wsi, model="pathprofiler", key_added="tissues_pathprofiler")
```

Store both under different keys and compare them visually.

## How do I perform cell segmentation?

Prepare tiles at the resolution expected by the model, then run:

```python
zs.pp.tile_tissues(wsi, 512, mpp=0.5)
zs.seg.cells(wsi, model="instanseg", key_added="cells")
zs.pl.annotations(wsi, key="cells")
```

Cell models are sensitive to MPP. NuLite and HistoPLUS support 20× or 40× modes and infer them from the tile specification when possible.

## How do I obtain cell types and features?

```python
zs.seg.cells(
    wsi,
    model="histoplus",
    extract_features=True,
    key_added="cells",
)
```

Compatible models add a class column to the cell shapes and store per-cell features as `cells_features`.

## How do I reduce cell-segmentation memory use?

```python
zs.seg.cells(
    wsi,
    model="histoplus",
    batch_size=1,
    num_workers=0,
    extract_features=True,
    low_memory=True,
)
```

Reduce batch size first. `low_memory=True` is particularly useful when extracting per-cell features.

## How do I handle overlapping segmentation tiles?

For cell segmentation, ownership filtering reduces duplicate polygonization before non-maximum suppression:

```python
zs.seg.cells(
    wsi,
    model="instanseg",
    overlap_ownership=True,
)
```

Use overlapping tiles only when the expected boundary benefit justifies the additional inference cost.

## How do I run semantic segmentation?

Instantiate a compatible segmentation model, prepare tiles at its expected input resolution, and pass the model object:

```python
zs.seg.semantic(
    wsi,
    model=model,
    tile_key="tiles",
    class_names=["background", "tumor", "stroma"],
    key_added="anatomical_structures",
)
```

The result is added to `wsi.shapes[key_added]`. Consult {func}`lazyslide.seg.semantic` for probability merging and threshold parameters.

## How do I evaluate segmentation?

Use semantic metrics such as Dice and mean IoU for class masks, and panoptic or instance statistics for individual objects. Ensure prediction and reference masks share the same resolution, extent, class mapping, and ignored-background definition before interpreting a score. See the [Metrics API](../api/metrics).
