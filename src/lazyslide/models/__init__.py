from lazyslide_models import (
    MODEL_REGISTRY,
    # Models
    DensePredictionModel,
    DensePredictionModelProtocol,
    ImageGenerationModel,
    ImageGenerationModelProtocol,
    ImageModel,
    ImageModelProtocol,
    ImageTextModel,
    ImageTextModelProtocol,
    MarkerMapModel,
    MarkerMapModelProtocol,
    ModelBase,
    ModelBaseProtocol,
    ModelTask,
    SegmentationModel,
    SegmentationModelProtocol,
    SlideEncoderModel,
    TilePredictionModel,
    TilePredictionModelProtocol,
    TimmModel,
    TimmViTModel,
    VirtualStainModel,
    VirtualStainModelProtocol,
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

from lazyslide._utils import warn_deprecated

warn_deprecated(
    "`lazyslide.models` is deprecated since v0.11.0 and will be removed in v0.14.0; "
    "install `lazyslide-models` and use `import lazyslide_models`."
)


def _register_compat_modules() -> None:
    """Mirror lazyslide-models modules into the legacy lazyslide.models namespace."""
    import sys

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
    "MODEL_REGISTRY",
    "DensePredictionModel",
    "DensePredictionModelProtocol",
    "ImageGenerationModel",
    "ImageGenerationModelProtocol",
    "ImageModel",
    "ImageModelProtocol",
    "ImageTextModel",
    "ImageTextModelProtocol",
    "MarkerMapModel",
    "MarkerMapModelProtocol",
    "ModelBase",
    "ModelBaseProtocol",
    "ModelTask",
    "SegmentationModel",
    "SegmentationModelProtocol",
    "SlideEncoderModel",
    "TilePredictionModel",
    "TilePredictionModelProtocol",
    "TimmModel",
    "TimmViTModel",
    "ViTModelProtocol",
    "VirtualStainModel",
    "VirtualStainModelProtocol",
    "base",
    "image_generation",
    "list_models",
    "multimodal",
    "register",
    "segmentation",
    "style_transfer",
    "tile_prediction",
    "vision",
]
