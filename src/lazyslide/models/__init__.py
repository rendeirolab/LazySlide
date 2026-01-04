from __future__ import annotations

from . import (
    image_generation,
    multimodal,
    segmentation,
    style_transfer,
    tile_prediction,
    vision,
)
from ._model_registry import MODEL_REGISTRY, register
from .base import (
    ImageModel,
    ImageTextModel,
    ModelBase,
    ModelTask,
    SegmentationModel,
    SlideEncoderModel,
    StyleTransferModel,
    TimmModel,
)

__all__ = [
    "multimodal",
    "segmentation",
    "style_transfer",
    "tile_prediction",
    "vision",
    "image_generation",
    "MODEL_REGISTRY",
    "register",
    "ImageModel",
    "ImageTextModel",
    "ModelBase",
    "ModelTask",
    "SegmentationModel",
    "SlideEncoderModel",
    "StyleTransferModel",
    "TimmModel",
]


def list_models(task: ModelTask | str = None):
    """List all available models.

    If you want to get models for feature extraction,
    you can use task='vision' or task='multimodal'.

    Parameters
    ----------
    task : {'vision', 'segmentation', 'multimodal', 'tile_prediction'}, default: None
        The task to filter the models. If None, return all models.

    Returns
    -------
    list
        A list of model names.

    """
    if task is None:
        return list(MODEL_REGISTRY.keys())
    else:
        try:
            task = ModelTask(task)
        except ValueError:
            raise ValueError(
                f"Unknown task: {task}. "
                f"Available tasks are: {', '.join([t.value for t in ModelTask])}."
            )
        models = []
        for name, model_cls in MODEL_REGISTRY.items():
            model_tasks = getattr(model_cls, "task", [])
            if isinstance(model_tasks, ModelTask):
                model_tasks = [model_tasks]
            if task in model_tasks:
                models.append(name)
        return models
