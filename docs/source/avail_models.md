# Model Zoo

This section provides an overview of models in LazySlide.

:::{important}
Disclaimer: The usage of any model in LazySlide is subject to the terms and conditions of the respective model's license.
Please ensure you comply with the license terms before using any model. If you use a model in your research, please
cite the original paper or repository as appropriate.
LazySlide does not redistribute any source code that's not compatible with LazySlide's MIT license.
:::

## Get model names

In most of the cases, you only need to pass the model name as string to the function, for example, to use
the `UNI` model in feature extraction, you can do: `zs.tl.feature_extraction(wsi, model="uni")`.

To get all available models, you can use the `list_models` function:

```python
from lazyslide.models import list_models

models = list_models()
```


You can also filter models by type:

```python
from lazyslide.models import list_models

vision_models = list_models("vision")  # for vision models only
multimodal_models = list_models("multimodal")  # for multimodal models only
segmentation_models = list_models("segmentation")  # for segmentation models only
tile_prediction_models = list_models("tile_prediction")  # for tile_prediction models only
```

For feature extraction, we also support all timm models with feature extraction head.
You can list them with:

```python
from timm import list_models

timm_models = list_models()
```

To retrive a specific model class:

```python
from lazyslide.models import MODEL_REGISTRY
model_module = MODEL_REGISTRY['instanseg']
model = model_module()  # Initiate the model
```

## Get access to gated models

{octicon}`check-circle-fill;1em;sd-text-success;` indicates the model is publicly available.
You can use it without a Hugging Face account or requesting access.

{octicon}`lock;1em;sd-text-danger;` indicates the model is gated and requires permission.
You must apply for access via the Hugging Face model card or the model's repository.

To access gated models, follow these steps:

1. Create a Hugging Face account: https://huggingface.co/

2. Visit the model card page and request access.
   You can also use the Hugging Face button provided for each model below.

3. In your account settings, go to Access Tokens and create a new token
   with the required permissions (read access is sufficient).
   This token will grant access to any new models you gain permission for in the future.

4. Log in using your token to access gated models.
   Run the following command:

   ```bash
   hf auth login --token YOUR_TOKEN
   ```

Below is a list of available models categorized by their type:

```{eval-rst}
.. include:: api/models.rst
```

## Use model in offline environment

For huggingface gated models, to run in an environment without internet access.
The model must be download first, for example, run the model initiation code on a HPC login node.

```python
from lazyslide.models import MODEL_REGISTRY

# This will cache the model
model = MODEL_REGISTRY['uni']()
```

When you submit a job to compute node without internet connection. Please set the environment variable
`HF_HUB_OFFLINE=1` so huggingface will not make any HTTP request.
Alternatively, You can set it at the start of your python session

```python
import os
os.environ['HF_HUB_OFFLINE'] = 1
```

How to use new models
---------------------

If you want to use a model that's not available in LazySlide, you can still use it by wrapping it with LazySlide's
model classes. If you are familiar with class inheritance, the following example should be quite easy for you.

Suppose you have a new vision model and you want to use it for feature extraction, you can simply inherit from
one of our base classes (here is `ImageModel`) and implement necessary methods.

```python
import torch

from lazyslide.models.base import ImageModel

class MyGreatModel(ImageModel):

    def __init__(self):
        from huggingface_hub import hf_hub_download

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

Once you finish with implementing your own model, you are welcomed to submit it to LazySlide.
Please take a look at [Contribution to new models](contributing/new_models.md)
