# Submission of new models

If you developed a new model or you find a new model that you want to add to LazySlide, here are the instructions.

## Submit an issue with `[New Model]`

Before you actually work on the new model, please submit an issue saying that you want to integrate
a new model and provide information on the model. We will then decide if that model should be integrated or not.
If the model is considered beneficial to LazySlide user, you can then start coding!

For setting up the development environment of LazySlide, please refer to [this page](setup.md).

## Understanding the base class of different model types

There are different types of models in LazySlide:
- Vision model
- Image-text multimodal model
- Segmentation model
- Tile prediction model

You should find all the base class definition in `src/lazyslide/models/base.py`, and all models should 
inherit from one of the base class. If you want the model like `model='cellpose'` in the LazySlide functions, 
please use the `register` decorator to register the model in LazySlide. 

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

Vision model will require to implement `encode_image(self, image)` method to encode the input image and return
the encoded feature.

Here is an example of a **vision model**:

You can have much information related to the model defined.
Some of them are required, some are optional. If it's optional, you don't have to define it.

If the model comes with a publication, please also add a bib entry in `docs/source/references.bib` and add the 
key to the `bib_key` field.

```python
import torch

from lazyslide.models import hf_access, register
from lazyslide.models.base import ImageModel, ModelTask


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

### Image-text multimodal model

Image text model will require implementing
- `encode_image(self, image)`: Same as vision model
- `encode_text(self, text)`: Tokenize the text, encode and normalize the text features

Here is an example of an **image-text multimodal model**:

```python
import torch

from lazyslide.models import hf_access, register
from lazyslide.models.base import ImageTextModel, ModelTask


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

Segmentation model will require implementing, depends on the model type, either semantics or instance segmentation, please
set the output type in `supported_output(self)` method.
- `segment(self, image)`: Segment the input image, must return a dictionary with the output of the model, the key should be the output type defined in `supported_output(self)`
- `supported_output(self)`: Return the supported output of the model, supported values are "probability_map", "instance_map", "class_map"

```python
import torch

from lazyslide.models.base import SegmentationModel, ModelTask
from lazyslide.models import register


@register(
    key="super-segmentation",
    task=ModelTask.segmentation,
    license="Apache 2.0",
    commercial=True,
)
class MySuperSegmentation(SegmentationModel):
    """Apply the InstaSeg model to the input image."""

    def __init__(self):
        from huggingface_hub import hf_hub_download

        model_file = hf_hub_download("my-repo/my-super-segmentation", "model.pt")

        self.model = torch.jit.load(model_file, map_location="cpu")
        self.model.eval()

    @torch.inference_mode()
    def segment(self, image):
        out = self.model(image)
        return {"instance_map": out.long().squeeze(1)}

    def supported_output(self):
        return ("instance_map",)  # Can be multiple outputs

```

### Tile prediction model

Tile prediction model will require implementing
- `predict(self, image)`: Return a dictionary that can be parsed as `DataFrame`, the values should have the same size as batch size.

```python
import torch

from lazyslide.models.base import TilePredictionModel, ModelTask
from lazyslide.models import register


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
from lazyslide.models import MODEL_REGISTRY, list_models

model_name = "your model name"
assert model_name in list_models()
model_class = MODEL_REGISTRY[model_name]
model_instance = model_class()  # You must call the model to initiate an instance

```
To make the model available to user, you will also need to go to the respective function to add your model logic.
If it's only for feature extraction, you don't need to do anything.

## Add a unit test

The final step is to ensure that your model be loaded and run properly. Please add a unit test to `tests/models/*.py`.
You can refer to the existing unit test to see how to write a unit test. If your model is gated on huggingface,
the model access must be granted to [@Mr-Milk](https://github.com/Mr-Milk) so that the test can be run on GitHub Actions.
Remember to @Mr-Milk when you submit a PR.

## Submit a PR

Now that you've done all the work, you can submit a PR and we will start the review process.
For instructions on how to submit a PR, 
please refer to [this page](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request).
