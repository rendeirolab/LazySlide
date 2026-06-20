# Troubleshooting

## A slide will not open

1. Confirm that the path is correct and readable.
2. Open the file with `zs.open_wsi(path)` in a minimal Python session to isolate reading from the analysis pipeline.
3. Install a reader that supports the format.
4. Test another file from the same scanner.
5. Check whether the file is truncated, encrypted, or stored on an unavailable mount.

See [supported readers and formats](../reference/formats).

## MPP is missing or wrong

Do not infer MPP solely from the filename or nominal magnification. Retrieve it from scanner metadata or acquisition records. When the source MPP is known, override it during tiling:

```python
zs.pp.tile_tissues(wsi, 256, mpp=0.5, slide_mpp=0.25)
```

Record the override in the analysis manifest.

## No tissue or no tiles are found

Visualize each intermediate result. Compare tissue methods, lower minimum-area thresholds, and confirm tile field of view. For tiling, temporarily disable background filtering to determine whether geometry or coverage filtering is responsible:

```python
zs.pp.tile_tissues(wsi, 256, mpp=0.5, background_filter=False)
```

## A model cannot be downloaded

Check the model identifier, internet access, Hugging Face authentication, gated-model approval, disk quota, and `HF_HOME`. On offline nodes, pre-populate the cache and set `HF_HUB_OFFLINE=1`.

## Inference runs out of memory

Reduce `batch_size`, then reduce `num_workers`. Enable mixed precision on supported hardware, select a smaller model, or process fewer tiles. Restart the process after an out-of-memory error if the framework has retained allocations.

## A key is not found

Inspect the object instead of guessing:

```python
print(list(wsi.shapes))
print(list(wsi.tables))
print(list(wsi.images))
```

Remember that feature tables usually include the tile key as a suffix, such as `uni_tiles`. Check the [output-key reference](../reference/outputs).

## Feature and tile row counts do not match

Feature rows correspond to the tile table used during extraction. This mismatch usually means that the tile table was filtered or replaced afterward, or the wrong `tile_key` was selected. Regenerate features from the current tiles or restore the matching tile table; do not truncate arrays to force equal lengths.

## Annotations are shifted, scaled, or mirrored

Confirm that coordinates are level-0 pixels, check slide bounds and MPP, and verify whether the source tool uses a bounded-image origin. Test `in_bounds` on a new key and compare several known landmarks. NDPA also requires valid Hamamatsu offsets and MPP metadata.

## A plot is blank

Plot the underlying shapes without feature coloring. Then verify the selected tissue, tile, feature, and color keys; zoom bounds; value range; and annotation coordinate system. Add options back one at a time.

## Results differ between runs

`find_tissues(level="auto")` can select a level based on available memory. Set a fixed level for repeatability. Also record package versions, model weight revisions, device, precision, random seeds, and all preprocessing parameters.
