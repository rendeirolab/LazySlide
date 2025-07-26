Model Zoo
=========

This section provides an overview of models in LazySlide.

.. important::

    Disclaimer: The usage of any model in LazySlide is subject to the terms and conditions of the respective model's license.
    Please ensure you comply with the license terms before using any model. If you use a model in your research, please
    cite the original paper or repository as appropriate.
    LazySlide do not redistribute any source code that's not compatible with LazySlide's MIT license.


Get model names
----------------

In most of the cases, you only need to pass the model name as string to the function, for example, to use
the `UNI` model in feature extraction, you can do: :code:`zs.tl.feature_extraction(wsi, model="uni")`.

To get all available models, you can use the `list_models` function:

.. code-block:: python

    from lazyslide.models import list_models

    models = list_models()


You can also filter models by type:

.. code-block:: python

    from lazyslide.models import list_models

    vision_models = list_models("vision")  # for vision models only
    multimodal_models = list_models("multimodal")  # for multimodal models only
    segmentation_models = list_models("segmentation")  # for segmentation models only
    tile_prediction_models = list_models("tile_prediction")  # for tile_prediction models only

For feature extraction, we also support all timm models with feature extraction head.
You can list them with:

.. code-block:: python

    from timm import list_models

    timm_models = list_models()


Get access to gated models
---------------------------

:octicon:`check-circle-fill;1em;sd-text-success;` indicates the model is publicly available.
You can use it without a Hugging Face account or requesting access.

:octicon:`lock;1em;sd-text-danger;` indicates the model is gated and requires permission.
You must apply for access via the Hugging Face model card or the model's repository.

To access gated models, follow these steps:

#. Create a Hugging Face account: https://huggingface.co/

#. Visit the model card page and request access.
   You can also use the Hugging Face button provided for each model below.

#. In your account settings, go to Access Tokens and create a new token
   with the required permissions (read access is sufficient).
   This token will grant access to any new models you gain permission for in the future.

#. Log in using your token to access gated models.
   Run the following command:

   .. code-block:: bash

      huggingface-cli login --token YOUR_TOKEN

Below is a list of available models categorized by their type:

Vision models
~~~~~~~~~~~~~~~~~~

.. currentmodule:: lazyslide.models.vision

.. autosummary::
    :nosignatures:

    UNI
    UNI2
    GigaPath
    PLIPVision
    CONCHVision
    Virchow
    Virchow2
    Phikon
    PhikonV2
    HOptimus0
    HOptimus1
    H0Mini
    HibouB
    HibouL


Multimodal Models
~~~~~~~~~~~~~~~~~

.. currentmodule:: lazyslide.models.multimodal

.. autosummary::
    :nosignatures:

    PLIP
    CONCH
    Titan
    Prism
    OmiCLIP


Segmentation Models
~~~~~~~~~~~~~~~~~~~

.. currentmodule:: lazyslide.models.segmentation

.. autosummary::
    :nosignatures:

    Instanseg
    NuLite
    GrandQCTissue
    GrandQCArtifact
    SMPBase

Base Models
~~~~~~~~~~~

.. currentmodule:: lazyslide.models.base

.. autosummary::
    :nosignatures:

    ModelBase
    ImageModel
    ImageTextModel
    SegmentationModel
    SlideEncoderModel
    TimmModel

