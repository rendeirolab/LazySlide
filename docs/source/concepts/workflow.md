# The LazySlide workflow

LazySlide operations are grouped by intent:

| Namespace | Purpose | Examples |
|---|---|---|
| `zs.pp` | Prepare spatial regions | find tissue, create tiles, build a graph |
| `zs.seg` | Run learned segmentation | tissue, cells, semantic classes, artifacts |
| `zs.tl` | Analyze tiles and features | embeddings, predictions, spatial domains |
| `zs.pl` | Inspect and visualize results | tissue, tiles, annotations, interactive viewer |
| `zs.io` | Exchange annotations | load and export spatial annotations |

## Dependencies between stages

Most pipelines follow this order:

1. Open a WSI and validate its metadata.
2. Find or import regions of interest.
3. Create one or more tile sets.
4. Optionally score or filter tile quality.
5. Run segmentation, feature extraction, or prediction.
6. Build spatial or slide-level summaries.
7. Visualize and save the result.

Later stages refer to earlier results by key. For example, feature extraction reads a `tile_key`, then creates a feature table associated with those exact tiles. If tiles are replaced afterward, the old feature rows may no longer correspond to the tile table.

## Functions add data in place

Many LazySlide functions mutate `wsi` and return `None`. This is intentional:

```python
zs.pp.find_tissues(wsi)                 # adds wsi.shapes["tissues"]
zs.pp.tile_tissues(wsi, 256, mpp=0.5) # adds wsi.shapes["tiles"]
zs.tl.feature_extraction(wsi, "uni")  # adds a feature table
```

Use `key_added` to preserve alternative results:

```python
zs.pp.find_tissues(wsi, method="entropy", key_added="tissues_entropy")
zs.pp.tile_tissues(
    wsi,
    256,
    mpp=1.0,
    tissue_key="tissues_entropy",
    key_added="tiles_1mpp",
)
```

See the [output-key reference](../reference/outputs) for the default locations.
