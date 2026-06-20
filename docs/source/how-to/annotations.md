# Annotations

## How do I import GeoJSON, QuPath, or NDPA annotations?

```python
zs.io.load_annotations(wsi, "annotations.geojson", key_added="annotations")
zs.io.load_annotations(wsi, "slide.ndpa", key_added="annotations")
```

GeoJSON exported by QuPath is read through GeoPandas. LazySlide also handles Hamamatsu NDPA coordinates using metadata from the matching slide.

## How do I inspect annotation classes and geometries?

```python
annotations = wsi.shapes["annotations"]
print(annotations.columns)
print(annotations.geom_type.value_counts())
print(annotations.head())
zs.pl.annotations(wsi, key="annotations")
```

Do this before filtering or joining. External tools can store class labels in nested fields; `json_flatten="classification"` handles the usual QuPath classification field.

## How do I join annotations to tiles or other shapes?

Use `join_with` for source layers and `join_to` for the destination layer:

```python
zs.io.load_annotations(
    wsi,
    "annotations.geojson",
    join_with="tiles",
    join_to="tiles",
    key_added="annotations",
)
```

Inspect the resulting columns to confirm the spatial relationship matches the intended labeling rule. Boundary tiles may intersect more than one annotation.

## How do I analyze only annotated regions?

Use the imported annotation key as the region source for tiling:

```python
zs.pp.tile_tissues(
    wsi,
    256,
    mpp=0.5,
    tissue_key="annotations",
    key_added="roi_tiles",
)
```

Filter the annotation `GeoDataFrame` first if only one class should be tiled, then store the filtered shapes under a new key.

## How do I fix shifted annotations?

Determine which coordinate system the exporter used. LazySlide shape coordinates are level-0 pixels. The `in_bounds=True` option translates annotations using slide bounds when the source coordinates are relative to the bounded image:

```python
zs.io.load_annotations(
    wsi,
    "annotations.geojson",
    in_bounds=True,
    key_added="annotations_in_bounds",
)
```

Do not apply the translation merely until a plot “looks close.” Verify against known landmarks and scanner metadata.

## How do I export annotations for QuPath?

```python
zs.io.export_annotations(
    wsi,
    key="cells",
    classes="class",
    format="qupath",
    file="cells.geojson",
)
```

The parent directory must already exist. Use `in_bounds=True` if the receiving tool expects coordinates translated out of the slide bounds.
