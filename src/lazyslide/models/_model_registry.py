from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Type, List

import pandas as pd

from . import ModelBase
from . import multimodal
from . import segmentation
from . import vision


class ModelTask(Enum):
    vision = "vision"
    segmentation = "segmentation"
    multimodal = "multimodal"


@dataclass
class ModelCard:
    name: str
    model_type: ModelTask
    module: Type[ModelBase]
    github_url: str = None
    hf_url: str = None
    paper_url: str = None
    description: str = None
    keys: List[str] = None

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
        skeleton = ""
        if self.github_url is not None:
            skeleton += f":octicon:`mark-github;1em;` `GitHub <{self.github_url}>`__ \\"
        if self.hf_url is not None:
            skeleton += f"🤗 `Hugging Face <{self.hf_url}>`__ \\"
        if self.paper_url is not None:
            skeleton += f" :octicon:`book;1em;` `Paper <{self.paper_url}>`__"
        if self.description is not None:
            skeleton += f"\n| {self.description}"

        return skeleton


MODEL_REGISTRY = {}

MODEL_DB = pd.read_csv(f"{Path(__file__).parent}/model_registry.csv")
_modules = {
    ModelTask.vision: vision,
    ModelTask.segmentation: segmentation,
    ModelTask.multimodal: multimodal,
}

for _, row in MODEL_DB.iterrows():
    model_type = ModelTask(row["model_type"])
    card = ModelCard(
        name=row["name"],
        model_type=model_type,
        module=getattr(_modules[model_type], row["module"]),
        github_url=None if pd.isna(row["github_url"]) else row["github_url"],
        hf_url=None if pd.isna(row["hf_url"]) else row["hf_url"],
        paper_url=None if pd.isna(row["paper_url"]) else row["paper_url"],
        description=None if pd.isna(row["description"]) else row["description"],
    )
    keys = [i.strip() for i in row["keys"].split(",")] if row["keys"] else []
    for key in keys:
        MODEL_REGISTRY[key] = card


def list_models(task: ModelTask = None):
    """List all available models.

    If you want to get models for feature extraction,
    you can use task='vision' or task='multimodal'.

    Parameters
    ----------
    task : {'vision', 'segmentation', 'multimodal'}, default: None
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
                "Available tasks are: vision, segmentation, multimodal."
            )
