import warnings

from lazyslide_models import (
    MODEL_REGISTRY,
    # Models
    ImageGenerationModel,
    ImageGenerationModelProtocol,
    ImageModel,
    ImageModelProtocol,
    ImageTextModel,
    ImageTextModelProtocol,
    ModelBase,
    ModelBaseProtocol,
    ModelTask,
    SegmentationModel,
    SegmentationModelProtocol,
    SlideEncoderModel,
    StyleTransferModel,
    StyleTransferModelProtocol,
    TilePredictionModelProtocol,
    TimmModel,
    TimmViTModel,
    ViTModelProtocol,
    base,
    image_generation,
    list_models,
    multimodal,
    register,
    segmentation,
    style_transfer,
    tile_prediction,
    vision,
)

warnings.warn(
    "Importing from 'lazyslide.models' is deprecated. "
    "Please use 'lazyslide-models' and use 'import lazyslide_models' instead. "
    "The models module will be removed in future versions.",
    category=FutureWarning,
    stacklevel=2,
)


def _register_compat_modules() -> None:
    import sys

    """Mirror lazyslide-models modules into the legacy lazyslide.models namespace."""
    sys.modules.update(
        {
            f"{__name__}.base": base,
            f"{__name__}.image_generation": image_generation,
            f"{__name__}.multimodal": multimodal,
            f"{__name__}.segmentation": segmentation,
            f"{__name__}.style_transfer": style_transfer,
            f"{__name__}.tile_prediction": tile_prediction,
            f"{__name__}.vision": vision,
        }
    )

    for source_name, module in tuple(sys.modules.items()):
        if source_name.startswith("lazyslide_models."):
            compat_name = source_name.replace("lazyslide_models", __name__, 1)
            sys.modules[compat_name] = module


_register_compat_modules()
del _register_compat_modules

__all__ = [
    "multimodal",
    "segmentation",
    "style_transfer",
    "tile_prediction",
    "vision",
    "image_generation",
    "base",
    "MODEL_REGISTRY",
    "register",
    "list_models",
    "ImageGenerationModel",
    "ImageGenerationModelProtocol",
    "ImageModel",
    "ImageModelProtocol",
    "ImageTextModel",
    "ImageTextModelProtocol",
    "ModelBase",
    "ModelBaseProtocol",
    "ModelTask",
    "SegmentationModel",
    "SegmentationModelProtocol",
    "SlideEncoderModel",
    "StyleTransferModel",
    "StyleTransferModelProtocol",
    "TilePredictionModelProtocol",
    "TimmModel",
    "TimmViTModel",
    "ViTModelProtocol",
]
