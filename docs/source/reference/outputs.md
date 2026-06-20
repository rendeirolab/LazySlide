# Outputs and key naming

Most LazySlide operations add results to `wsi`. Defaults can be changed with `key_added`.

| Operation | Default result | Location |
|---|---|---|
| `pp.find_tissues` | `tissues` | `wsi.shapes` |
| `pp.tile_tissues` | `tiles` | `wsi.shapes` |
| `seg.tissue` | `tissues` | `wsi.shapes` |
| `seg.cells` | `cells` | `wsi.shapes` |
| `seg.cells(..., extract_features=True)` | `cells_features` | `wsi.tables` |
| `seg.semantic` | `anatomical_structures` | `wsi.shapes` |
| `io.load_annotations` | `annotations` | `wsi.shapes` |
| `tl.feature_extraction` | `{model}_{tile_key}` | `wsi.tables` |
| `tl.feature_aggregation` | `agg_slide` or `agg_{by}` | inside feature `AnnData` |

## Resolve keys safely

```python
print(list(wsi.shapes))
print(list(wsi.tables))
print(list(wsi.images))
print(list(wsi.attrs))
```

Many APIs accept a short feature key and add the tile suffix automatically:

```python
zs.tl.feature_extraction(wsi, "uni")
zs.pl.tiles(wsi, feature_key="uni", color="0")
```

The stored table is normally `uni_tiles`. With a custom tile set, it becomes `uni_{tile_key}`.

## Recommended naming

Encode the meaningful variant, not every parameter:

```python
tissue_key = "tissues_entropy"
tile_key = "tiles_20x"
feature_name = "uni_experiment_a"
```

Keep the complete parameter set in a machine-readable run configuration. Keys should remain short enough to pass between functions without mistakes.
