from typing import Dict, Type

from . import multimodal
from . import segmentation
from . import vision
from .base import (
    ModelBase,
    ImageModel,
    ImageTextModel,
    SegmentationModel,
    SlideEncoderModel,
    TimmModel,
)

from ._model_registry import MODEL_REGISTRY, list_models
