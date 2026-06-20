# Choosing a model

Choose a model by working backward from the output you need. A newer or larger model is not automatically the best fit.

## Decision checklist

1. **Task:** tissue segmentation, cell segmentation, tile embedding, tile prediction, image-text analysis, slide encoding, or image generation.
2. **Input contract:** tile size, MPP or magnification, stain, and supported tissue domain.
3. **Output contract:** polygons, class scores, tile embeddings, dense tokens, or a slide vector.
4. **Access:** public weights, gated Hugging Face access, or an additional package.
5. **Compute:** CPU feasibility, GPU memory, model size, and expected tile count.
6. **License and citation:** whether the model is permitted for the intended use.

List registered models by task:

```python
from lazyslide_models import list_models

list_models("vision")
list_models("segmentation")
list_models("multimodal")
list_models("tile_prediction")
```

The [Model Zoo](../avail_models) records available models and access requirements. Individual model API pages document their parameters.

## Start with a baseline

Use a small, public model to validate the pipeline before downloading gated foundation-model weights. This separates workflow problems—incorrect MPP, empty tiles, output keys—from model-access and memory problems.

## Preserve the input specification

Record the model name, package version, requested MPP, tile size, transform, pooling mode, and aggregation method. Embeddings produced under different input specifications should not be treated as interchangeable merely because they have the same dimension.
