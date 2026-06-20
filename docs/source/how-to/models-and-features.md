# Models and features

## How do I list models for a task?

```python
from lazyslide_models import list_models

vision_models = list_models("vision")
cell_models = list_models("segmentation")
multimodal_models = list_models("multimodal")
```

Use the [Model Zoo](../avail_models) to compare access and model classes, then inspect the individual API page for its input assumptions.

## How do I choose a feature extractor?

Start from the intended downstream task, tissue domain, input MPP, compute budget, and licensing constraints. Validate the pipeline with a small public model before moving to a gated foundation model. See [Choosing a model](../concepts/choosing-models).

```python
zs.tl.feature_extraction(wsi, model="resnet50")
```

## How do I select CPU, CUDA, or Apple MPS?

Pass the device explicitly when reproducibility matters:

```python
zs.tl.feature_extraction(wsi, "uni", device="cuda")
zs.tl.feature_extraction(wsi, "resnet50", device="cpu")
zs.tl.feature_extraction(wsi, "uni", device="mps")
```

Automatic selection is convenient for exploration but can make runtime behavior vary between machines.

## How do I reduce memory use?

Reduce `batch_size` first. Then consider fewer workers, mixed precision on supported hardware, a smaller model, fewer tiles, or a coarser MPP where scientifically appropriate.

```python
zs.tl.feature_extraction(
    wsi,
    "uni",
    device="cuda",
    batch_size=8,
    num_workers=2,
    amp=True,
)
```

Increasing `num_workers` can improve input throughput but also increases host-memory use. Tune it separately from batch size.

## Where are extracted features stored?

The default table key is `{model_name}_{tile_key}`. With model `uni` and tile key `tiles`, it is normally `uni_tiles`:

```python
zs.tl.feature_extraction(wsi, "uni")
features = wsi.tables["uni_tiles"]
print(features.shape)
```

LazySlide accepts the short model name in many `feature_key` arguments and resolves the tile suffix. Use the full table key while inspecting storage.

## How do I control the output key?

```python
zs.tl.feature_extraction(
    wsi,
    model="uni",
    tile_key="tiles_20x",
    key_added="uni_experiment_a",
)
```

The tile key is still added as a suffix, so inspect `wsi.tables.keys()` after the call. Use descriptive keys in pipelines that compare model or preprocessing variants.

## How do I extract dense patch-token features?

Dense extraction is available only for compatible ViT models:

```python
zs.tl.feature_extraction(wsi, model="uni", dense=True)
```

This creates tile-level features plus a denser tile grid associated with patch tokens. It uses substantially more storage than one vector per input tile.

## How do I control ViT pooling?

```python
zs.tl.feature_extraction(
    wsi,
    model="uni",
    dense=True,
    pool_mode="cls_patch_mean",
)
```

- `cls` uses the CLS token.
- `cls_patch_mean` concatenates the CLS token and the mean patch token.

Pooling changes feature dimensionality and semantics. Record it as part of the analysis specification.

## How do I extract features for cells?

Use the cell shape key as the image-region key:

```python
zs.tl.feature_extraction(wsi, model="uni", tile_key="cells")
```

Compatible ViT cell-segmentation models can produce segmentation and per-cell features together:

```python
zs.seg.cells(wsi, model="histoplus", extract_features=True)
```

The combined workflow stores cells under `cells` and their features under `cells_features`.

## How do I aggregate features by tissue or slide?

```python
# One representation per tissue piece
zs.tl.feature_aggregation(
    wsi,
    feature_key="uni",
    by="tissue_id",
)

# One representation for the complete slide
zs.tl.feature_aggregation(wsi, feature_key="uni", encoder="mean")
```

Aggregation metadata and results are stored inside the feature `AnnData`; they are not a new shape layer. See the {func}`lazyslide.tl.feature_aggregation` API for learned slide encoders and exact output locations.
