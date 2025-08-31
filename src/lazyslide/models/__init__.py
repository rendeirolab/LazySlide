from __future__ import annotations

from . import multimodal, segmentation, tile_prediction, vision
from ._utils import hf_access
from .base import (
    ImageModel,
    ImageTextModel,
    ModelBase,
    ModelTask,
    SegmentationModel,
    SlideEncoderModel,
    TimmModel,
)

MODEL_REGISTRY = ModelBase.registry


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
    if task is not None:
        try:
            task = ModelTask(task)
        except ValueError:
            raise ValueError(
                f"Unknown task: {task}. "
                f"Available tasks are: {', '.join([t.value for t in ModelTask])}."
            )
        models = []
        for name, model in MODEL_REGISTRY.items():
            model_task = model.task
            if isinstance(model_task, ModelTask):
                model_task = [model_task]
            if task in model_task:
                models.append(name)
        return models
