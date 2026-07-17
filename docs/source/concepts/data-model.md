# How LazySlide stores results

LazySlide uses `WSIData` as the analysis container. The original WSI remains the pixel source; analysis results are organized into spatial shapes, tables, images, and metadata.

## The important collections

| Collection | Typical content |
|---|---|
| `wsi.shapes` | tissue polygons, tiles, cells, annotations |
| `wsi.tables` | feature embeddings and other annotated matrices |
| `wsi.images` | raster outputs such as generated or segmentation images |
| `wsi.attrs` | specifications and metadata describing results |

Inspect the object before guessing a key:

```python
print(wsi.shapes.keys())
print(wsi.tables.keys())
print(wsi.images.keys())
print(wsi.attrs.keys())
```

The shorthand `wsi["key"]` retrieves a named element, but explicit collections are clearer while debugging.

## Spatial coordinates

Shape geometries are expressed in the level-0 image coordinate system. LazySlide records how a tile set should be read at its requested MPP; the polygons still align with the full-resolution slide.

## Associations matter

A feature table is associated with the shape table that supplied its image regions. For default tiles, a model named `uni` normally creates `uni_tiles`. For a custom tile set:

```python
zs.pp.tile_tissues(wsi, 256, mpp=1.0, key_added="coarse_tiles")
zs.tl.feature_extraction(wsi, "uni", tile_key="coarse_tiles")
# Default feature key: uni_coarse_tiles
```

Pass `tile_key`, `feature_key`, and `key_added` explicitly in reusable pipelines. This makes dependencies visible and avoids accidentally reading a similarly named result.

## Persistence

Open a source WSI with a backing store when results must survive the Python session:

```python
wsi = zs.open_wsi("slide.svs", backed_file="slide-analysis.zarr")
# ...add results...
wsi.write(overwrite=True)
```

The backing store does not replace the source slide. Keep the source path valid when reopening an analysis that still needs to fetch image pixels.
