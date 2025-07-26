import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Type

from . import multimodal, segmentation, tile_prediction, vision
from .base import ModelBase


class ModelTask(Enum):
    vision = "vision"
    segmentation = "segmentation"
    multimodal = "multimodal"
    tile_prediction = "tile_prediction"


@dataclass
class ModelCard:
    name: str
    is_gated: bool
    model_type: ModelTask
    module: Type[ModelBase]
    github_url: str = None
    hf_url: str = None
    paper_url: str = None
    description: str = None
    keys: List[str] = None
    bib_key: str = None

    def __post_init__(self):
        try:
            inject_doc = str(self)
            origin_doc = self.module.__doc__
            if origin_doc is None:
                origin_doc = ""
            else:
                origin_doc = f"\n\n{origin_doc}"
            self.module.__doc__ = f"{inject_doc}{origin_doc}"
        except AttributeError:
            # If the module does not have a __doc__ attribute, skip the injection
            pass

        if self.keys is None:
            self.keys = [self.name.lower()]

    def __str__(self):
        skeleton = (
            ":octicon:`lock;1em;sd-text-danger;` "
            if self.is_gated
            else ":octicon:`check-circle-fill;1em;sd-text-success;` "
        )
        if self.hf_url is not None:
            skeleton += f":bdg-link-primary-line:`🤗Hugging Face <{self.hf_url}>` "
        if self.github_url is not None:
            skeleton += f":bdg-link-primary-line:`GitHub <{self.github_url}>` "
        if self.paper_url is not None:
            skeleton += f":bdg-link-primary-line:`Paper <{self.paper_url}>` "
        if self.bib_key is not None:
            skeleton += f":cite:p:`{self.bib_key}`"
        if self.description is not None:
            skeleton += f" {self.description}"

        return skeleton


MODEL_REGISTRY = {}

with open(f"{Path(__file__).parent}/model_registry.json", "r") as f:
    MODEL_DB = json.load(f)
_modules = {
    ModelTask.vision: vision,
    ModelTask.segmentation: segmentation,
    ModelTask.multimodal: multimodal,
    ModelTask.tile_prediction: tile_prediction,
}

for row in MODEL_DB:
    model_type = ModelTask(row["model_type"])
    card = ModelCard(
        name=row["name"],
        is_gated=row["is_gated"],
        model_type=model_type,
        module=getattr(_modules[model_type], row["module"]),
        github_url=row["github_url"],
        hf_url=row["hf_url"],
        paper_url=row["paper_url"],
        description=row["description"],
        bib_key=row.get("bib_key"),
    )
    keys = row["keys"]
    for key in keys:
        MODEL_REGISTRY[key] = card


def list_models(task: ModelTask = None):
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
        task = ModelTask(task)
        if task in ModelTask:
            return [
                name
                for name, model in MODEL_REGISTRY.items()
                if model.model_type == task
            ]
        else:
            raise ValueError(
                f"Unknown task: {task}. "
                "Available tasks are: vision, segmentation, multimodal and tile_prediction."
            )
