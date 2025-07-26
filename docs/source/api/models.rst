.. _models-section:

Models
------

.. currentmodule:: lazyslide.models

.. autosummary::
    :toctree: _autogen
    :nosignatures:

    list_models


Vision Models
~~~~~~~~~~~~~

.. currentmodule:: lazyslide.models.vision

.. autosummary::
    :toctree: _autogen
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
    :toctree: _autogen
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
    :toctree: _autogen
    :nosignatures:

    Instanseg
    NuLite
    GrandQCTissue
    GrandQCArtifact
    PathProfilerTissueSegmentation
    SMPBase

Tile Prediction Models
~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: lazyslide.models.tile_prediction

.. autosummary::
    :toctree: _autogen
    :nosignatures:

    SpiderBreast
    SpiderColorectal
    SpiderSkin
    SpiderThorax


Tile Prediction Models (Computer vision features)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These models are based on OpenCV but provided with a model inferface.

.. currentmodule:: lazyslide.models.tile_prediction

.. autosummary::
    :toctree: _autogen
    :nosignatures:

    Brightness
    Canny
    Contrast
    Entropy
    HaralickTexture
    Saturation
    Sharpness
    Sobel
    SplitRGB


Base Models
~~~~~~~~~~~

.. currentmodule:: lazyslide.models.base

.. autosummary::
    :toctree: _autogen
    :nosignatures:

    ModelBase
    ImageModel
    ImageTextModel
    SegmentationModel
    SlideEncoderModel
    TilePredictionModel
    TimmModel
