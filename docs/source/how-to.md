# How to?

This section provides useful tips when using LazySlide.

## Tissue segmentation

:::{dropdown} My tissues look very vague, how do I segment them?

You can either try the entropy-based segmentation:
```python
zs.pp.find_tissues(wsi, method="entropy")
```

Or you can try the deep learning-based segmentation:
```python
zs.seg.tissue(wsi)
```
:::

## Feature extraction

:::{dropdown} How do I extract dense/patch features?

You can simply set `dense=True` when running feature extraction, 
notice that this only works for ViT-based models:
```python
zs.tl.feature_extraction(wsi, model="uni", dense=True)
```
:::

:::{dropdown} How do I control the pooling behavior of the extracted features?

There are currently two options for pooling the extracted features: "cls" and "cls_patch_mean". 
But this only works for ViT-based models.
- "cls": This option uses the [CLS] token as the pooled representation of the tissue tile. 
  Transformer-based models use the [CLS] token for classification tasks.
- "cls_patch_mean": This option concatenates the [CLS] token and the mean of patch tokens.
  This combines both global context (the CLS token) and local details (the patch tokens).
```python
zs.tl.feature_extraction(wsi, model="uni", pool_mode="cls_patch_mean")
```
:::

:::{dropdown} How do I extract features for cells?

Simply point the `tile_key` to "cells" when running feature extraction.
```python
zs.tl.feature_extraction(wsi, model="uni", tile_key="cells")
```

For ViT-based cell segmentation models, you can get both the cell segmentation and cell features at the same time.
```python
zs.seg.cell_types(wsi, model="histoplus", extract_features=True)
```
:::