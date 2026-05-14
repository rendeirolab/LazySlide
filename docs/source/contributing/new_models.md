# Submission of new models

If you developed a new model or you find a new model that you want to add to LazySlide, here are the instructions.

## Submit an issue with `[New Model]`

Before you actually work on the new model, please submit an issue saying that you want to integrate
a new model and provide information on the model. We will then decide if that model should be integrated or not.
If the model is considered beneficial to LazySlide user, you can then start coding!

As of LazySlide v0.11.0, all models live in a separate package called
[lazyslide-models](https://github.com/rendeirolab/lazyslide-models).
New model contributions should be submitted to that repository.

## Understanding the base class of different model types

There are different types of models in LazySlide:
- {term}`Vision models <vision model>`
- Image-text {term}`multimodal models <multimodal model>`
- {term}`Segmentation models <segmentation model>`
- {term}`Tile prediction models <tile prediction model>`

You should find all the base class definitions in `lazyslide_models/base.py`, and all models should
inherit from one of the base classes. If you want the model to be usable like `model='cellpose'` in LazySlide functions,
please use the `register` decorator to register the model.

```python
zs.seg.cells(wsi, model='cellpose')  # 'cellpose' must be registered
```

Otherwise, you can simply pass the model instance as parameter.

```python
cellpose_instance = Cellpose()

zs.seg.cells(wsi, model=cellpose_instance)
```
    

A key field is required to defined the name of the model.
Please refer to the actual model definition to see how a model class is constructed.
Different model types will require different methods to be implemented.

There are some shared methods

- `get_transform(self)`: Return the transformation that should be applied to the input image.

### Vision model

Vision models must implement `encode_image(self, image)` to encode the input image and return
the encoded feature.

Here is an example of a **vision model**:

You can have much information related to the model defined.
Some of them are required, some are optional. If it's optional, you don't have to define it.

If the model comes with a publication, please also add a bib entry in `docs/source/references.bib` and add the 
key to the `bib_key` field.

```python
import torch

from lazyslide_models import hf_access, register
from lazyslide_models.base import ImageModel, ModelTask


# A key must be defined to register the model
@register(
    key="the key of the model",
    is_gated=False,  # Optional, by default is False, is the model gated on huggingface?
    task=ModelTask.vision,  # Required, can be a list
    license="MIT",  # Required
    description="The greatest model ever",
    commercial=True,  # Required, can the model be used for commercial purpose?
    hf_url="https://huggingface.co/xxx",  # Optional
    github_url="https://github.com/xxx/xxx",  # Optional
    paper_url="https://doi.org/xxx/xxx",  # Optional
    bib_key="xxx",  # Optional, Add the bib entry to docs/source/references.bib
    param_size="87.8M",  # Optional
    encode_dim=512,  # Optional
)
class MyGreatModel(ImageModel):

    def __init__(self):
        from huggingface_hub import hf_hub_download

        # Use this context manager if your model is gated on huggingface
        with hf_access("my-repo/my-great-model"):
            model_file = hf_hub_download("my-repo/my-great-model", "model.pt")

        self.model = torch.jit.load(model_file, map_location="cpu")
        self.model.eval()

    # Define the transformation here, will automatically be applied
    def get_transform(self):
        return self.model.get_custom_transform()

    @torch.inference_mode()
    def encode_image(self, image):
        """
        Encode the input image using the model.
        The model should expect a tensor of shape [B, C, H, W].
        """
        output = self.model(image)
        return output
```

### ViT vision model (with dense tokens)

If your vision model is a Vision Transformer (ViT), you should also implement `encode_image_dense(self, image)`
which returns a `DenseTokens` named tuple containing separate CLS and patch token embeddings.
This enables LazySlide to extract dense (per-patch) features via `dense=True` in `feature_extraction`.

`DenseTokens` fields:
- `cls_token`: CLS token embedding, shape `[B, D]`
- `patch_tokens`: Patch token embeddings, shape `[B, N_patches, D]`

If your model is a timm ViT, you can inherit from `TimmViTModel` which provides `encode_image_dense` automatically.
For non-timm ViTs, implement it yourself and import `DenseTokens` from `lazyslide_models.base`.

**Pooling behavior**: When `dense=True`, LazySlide pools the dense tokens into tile-level features.
The pool mode is controlled by `pool_mode` in `feature_extraction`:
- `"cls"`: Use only the CLS token (default for most models)
- `"cls_patch_mean"`: Concatenate CLS token with the mean of patch tokens

If your model's `encode_image` returns `cat(cls_token, patch_tokens.mean())`, you should add an entry
to `DEFAULT_POOL_MODE` in `src/lazyslide/tools/_features.py` with `"cls_patch_mean"` so the dense path
matches the non-dense behavior.

```python
import torch

from lazyslide_models import register
from lazyslide_models.base import DenseTokens, ImageModel, ModelTask


@register(
    key="my-great-vit",
    task=ModelTask.vision,
    license="MIT",
    description="A great ViT model",
    commercial=True,
)
class MyGreatViT(ImageModel):

    def __init__(self):
        from huggingface_hub import hf_hub_download

        model_file = hf_hub_download("my-repo/my-great-vit", "model.pt")
        self.model = torch.load(model_file, map_location="cpu")
        self.model.eval()
        self.num_prefix_tokens = 1  # typically 1 for CLS token

    @torch.inference_mode()
    def encode_image_dense(self, image) -> DenseTokens:
        hidden = self.model.forward_features(image)
        return DenseTokens(
            cls_token=hidden[:, 0],
            patch_tokens=hidden[:, self.num_prefix_tokens:],
        )

    @torch.inference_mode()
    def encode_image(self, image):
        dense = self.encode_image_dense(image)
        # CLS + mean patch tokens — add "my-great-vit": "cls_patch_mean" to DEFAULT_POOL_MODE
        return torch.cat([dense.cls_token, dense.patch_tokens.mean(dim=1)], dim=-1)
```

For timm-based ViTs, you can skip the manual implementation entirely:

```python
from lazyslide_models.base import TimmViTModel, ModelTask
from lazyslide_models import register


@register(
    key="my-timm-vit",
    task=ModelTask.vision,
    license="MIT",
    description="A timm ViT model",
    commercial=True,
)
class MyTimmViT(TimmViTModel):
    """Inherits encode_image, encode_image_dense, get_transform from TimmViTModel."""

    def __init__(self, token=None):
        super().__init__("hf-hub:my-repo/my-timm-vit", pretrained=True, token=token)
```

### Image-text multimodal model

Image text model will require implementing
- `encode_image(self, image)`: Same as vision models
- `encode_text(self, text)`: Tokenize the text, encode and normalize the text features

Here is an example of an **image-text multimodal model**:

```python
import torch

from lazyslide_models import hf_access, register
from lazyslide_models.base import ImageTextModel, ModelTask


@register(
    key="the key of the model",
    task=ModelTask.vision,  # Required, can be a list
    license="MIT",  # Required
    description="The greatest model ever",
    commercial=True,  # Required, can the model be used for commercial purpose?
)
class MyGreatImageTextModel(ImageTextModel):

    def __init__(self):
        from huggingface_hub import hf_hub_download

        # Use this context manager if your model is gated on huggingface
        with hf_access("my-repo/my-great-image-text-model"):
            model_file = hf_hub_download("my-repo/my-great-image-text-model", "model.pt")

        self.model = torch.jit.load(model_file, map_location="cpu")
        self.model.eval()

    @torch.inference_mode()
    def encode_image(self, image):
        return self.model.encode_image(image)

    @torch.inference_mode()
    def encode_text(self, text):
        return self.model.encode_text(text, normalize=True)

```

### Segmentation model

Segmentation models must implement `segment(self, image)` which returns a `SegmentationOutput` named tuple.
The output covers both semantic and instance segmentation — set the relevant fields and leave others as `None`.

- `segment(self, image)`: Segment the input image, must return a `SegmentationOutput`

`SegmentationOutput` fields:
- `probability_map`: Per-class probabilities, shape `[B, C, H, W]` (float). For semantic segmentation and cell type classification.
- `instance_map`: Instance ID map, shape `[B, H, W]` (int). For instance segmentation.
- `patch_token_map`: Vision token map, shape `[B, D, Patch_H, Patch_W]`. Only for ViT-based segmentation models.
- `classes`: Tuple of class name strings in index order. Set when the model has named classes.

```python
import torch

from lazyslide_models.base import SegmentationModel, SegmentationOutput, ModelTask
from lazyslide_models import register


@register(
    key="super-segmentation",
    task=ModelTask.segmentation,
    license="Apache 2.0",
    commercial=True,
)
class MySuperSegmentation(SegmentationModel):
    """Instance segmentation model."""

    def __init__(self):
        from huggingface_hub import hf_hub_download

        model_file = hf_hub_download("my-repo/my-super-segmentation", "model.pt")

        self.model = torch.jit.load(model_file, map_location="cpu")
        self.model.eval()

    @torch.inference_mode()
    def segment(self, image):
        out = self.model(image)
        return SegmentationOutput(instance_map=out.long().squeeze(1))

```

### Tile prediction model

Tile prediction model will require implementing
- `predict(self, image)`: Return a dictionary that can be parsed as `DataFrame`, the values should have the same size as batch size.

```python
import torch

from lazyslide_models.base import TilePredictionModel, ModelTask
from lazyslide_models import register


@register(
    key="super-tile-prediction",
    ... # Other Parameters
)
class MySuperTilePrediction(TilePredictionModel, ):
    
    def __init__(self):
        from huggingface_hub import hf_hub_download
        
        model_file = hf_hub_download("my-repo/my-super-tile-prediction", "model.pt")
        
        self.model = torch.jit.load(model_file, map_location="cpu")
        self.model.eval()
    
    @torch.inference_mode()
    def predict(self, image):
        out = self.model(image)
        return {
            "is_tumor": out["is_tumor"],
            "is_normal": out["is_normal"]
        }

```

## Test your model

If you've done everything right, your model should be available in the registry. Try to run the following code
to see if it works

```python
from lazyslide_models import MODEL_REGISTRY, list_models

model_name = "your model name"
assert model_name in list_models()
model_class = MODEL_REGISTRY[model_name]
model_instance = model_class()  # You must call the model to initiate an instance

```
To make the model available to users, you will also need to go to the respective function in the LazySlide
repository to add your model logic. If it's only for feature extraction, you don't need to do anything.

## Add a unit test

The final step is to ensure that your model can be loaded and run properly. Please add a unit test to the
[lazyslide-models](https://github.com/rendeirolab/lazyslide-models) repository under `tests/test_models.py`.
You can refer to the existing tests to see how to write one. If your model is gated on Hugging Face,
the model access must be granted to [@Mr-Milk](https://github.com/Mr-Milk) so that the test can be run on GitHub Actions.
Remember to @Mr-Milk when you submit a PR.

## Submit a PR

New models should be submitted as a PR to the
[lazyslide-models](https://github.com/rendeirolab/lazyslide-models) repository (not the main LazySlide repo).
For instructions on how to submit a PR, 
please refer to [this page](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request).
