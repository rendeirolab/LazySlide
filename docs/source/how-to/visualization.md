# Visualization

## How do I visualize tissue, tiles, and annotations?

```python
zs.pl.tissue(wsi)
zs.pl.tiles(wsi, linewidth=0.5)
zs.pl.annotations(wsi, key="annotations", color="class")
```

Use the same keys that generated the result:

```python
zs.pl.tissue(wsi, tissue_key="tissues_entropy")
zs.pl.tiles(wsi, tile_key="tiles_20x")
```

## How do I color tiles by a stored column?

When the value is a column in the tile shape table, pass `color` without `feature_key`:

```python
zs.pl.tiles(wsi, color="tissue_id", palette="tab10")
```

For an embedding or prediction table, provide its feature key:

```python
zs.pl.tiles(wsi, feature_key="uni", color="0", cmap="viridis")
```

Inspect the table's `var_names` or columns if you do not know the available color names.

## How do I focus on one tissue or region?

```python
zs.pl.tissue(wsi, tissue_id=0)
zs.pl.tiles(wsi, tissue_id=0)
```

Or pass a viewport in level-0 coordinates:

```python
zs.pl.tiles(wsi, zoom=(10_000, 20_000, 5_000, 15_000))
```

## How do I create a publication-quality figure?

Provide an axis and match `target_dpi` to the export DPI:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 6))
zs.pl.tiles(
    wsi,
    feature_key="uni",
    color="0",
    target_dpi=300,
    ax=ax,
)
fig.savefig("feature-map.png", dpi=300, bbox_inches="tight")
```

## Why is my plot blank or misaligned?

Check that:

- the requested shape and feature keys exist;
- the feature table is associated with the selected `tile_key`;
- imported annotations use level-0 pixel coordinates;
- `in_bounds` matches whether geometries include slide-bound offsets;
- the selected `tissue_id`, zoom window, or value range contains data.

Start by plotting shapes without feature coloring, then add one option at a time.
