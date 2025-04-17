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

VISION_MODEL_REGISTRY = {
    # Vision models
    "conch_vision": vision.CONCHVision,
    "gigapath": vision.GigaPath,
    "plip_vision": vision.PLIPVision,
    "uni": vision.UNI,
    "uni2": vision.UNI2,
    "virchow": vision.Virchow,
    "virchow2": vision.Virchow2,
    "phikon": vision.Phikon,
    "phikon-v2": vision.PhikonV2,
    "h-optimus-0": vision.HOptimus0,
    "h-optimus-1": vision.HOptimus1,
    "h0-mini": vision.H0Mini,
    "titan": multimodal.Titan,
    "conch_v1.5": multimodal.Titan,
}

SEGMENTATION_MODEL_REGISTRY = {
    # Segmentation models
    "nulite": segmentation.NuLite,
    "instanseg": segmentation.Instanseg,
}


MULTIMODAL_MODEL_REGISTRY = {
    "plip": multimodal.PLIP,
    "conch": multimodal.CONCH,
    "prism": multimodal.Prism,
    "titan": multimodal.Titan,
}


def list_models(task=None) -> Dict[str, Type[ModelBase]]:
    """List all available models."""
    if task is None:
        return {
            **VISION_MODEL_REGISTRY,
            **SEGMENTATION_MODEL_REGISTRY,
            **MULTIMODAL_MODEL_REGISTRY,
        }
    elif task == "vision":
        return VISION_MODEL_REGISTRY
    elif task == "segmentation":
        return SEGMENTATION_MODEL_REGISTRY
    elif task == "multimodal":
        return MULTIMODAL_MODEL_REGISTRY
    else:
        raise ValueError(
            f"Unknown task: {task}. "
            f"Supported tasks are: vision, segmentation, multimodal."
        )
