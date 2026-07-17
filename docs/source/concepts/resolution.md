# Resolution, magnification, and tiling

Resolution choices determine both what a model can see and how much computation a workflow requires.

## MPP and magnification

MPP is a physical measurement; magnification is scanner and optics terminology. Common approximations are:

| Nominal magnification | Approximate MPP |
|---:|---:|
| 10× | 1.0 |
| 20× | 0.5 |
| 40× | 0.25 |

These are conventions, not a substitute for metadata. Inspect the slide before assuming a value:

```python
print(wsi.properties)
print(wsi.fetch.pyramids())
```

If MPP is missing, provide the known source resolution through `slide_mpp` when tiling:

```python
zs.pp.tile_tissues(wsi, 256, mpp=0.5, slide_mpp=0.25)
```

Do this only when the scanner or acquisition record provides a trustworthy value.

## Field of view

The physical width of a square tile is:

```text
field of view in microns = tile_px * mpp
```

| Tile | MPP | Field of view |
|---:|---:|---:|
| 256 px | 0.5 | 128 µm |
| 512 px | 0.5 | 256 µm |
| 256 px | 0.25 | 64 µm |

Changing tile pixels and changing MPP are therefore not equivalent. Choose the input expected by the model whenever using a pretrained model.

## Stride and overlap

With no `stride_px` or `overlap`, adjacent tiles do not overlap. Smaller strides improve spatial coverage but increase tile count and compute.

```python
# 25% overlap
zs.pp.tile_tissues(wsi, 256, mpp=0.5, overlap=0.25)

# Equivalent explicit stride for square 256-pixel tiles
zs.pp.tile_tissues(wsi, 256, mpp=0.5, stride_px=192)
```

`overlap` and `stride_px` are alternatives; specify one, not both.

## Pyramid levels

Use `level` for operations that work directly on a stored pyramid level, such as traditional tissue detection. Use MPP for algorithms whose behavior must be physically comparable across scanners.
