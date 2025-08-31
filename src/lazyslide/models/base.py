from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, List

import torch

from ._model_registry import ModelRegistry
from ._repr import model_doc, model_repr_html
from ._utils import get_default_transform, hf_access


class ModelTask(Enum):
    vision = "vision"
    segmentation = "segmentation"
    multimodal = "multimodal"
    slide_encoder = "slide_encoder"
    tile_prediction = "tile_prediction"
    feature_prediction = "feature_prediction"
    style_transfer = "style_transfer"
    cv_feature = "cv_feature"


class ModelBase(ABC):
    model: Any
    task: ModelTask | List[ModelTask]
    is_gated: bool = False
    description: str = None
    github_url: str = None
    hf_url: str = None
    paper_url: str = None
    license: str = (None,)
    license_url: str = None
    bib_key: str = None
    commercial: bool = None
    param_size: int | str = None
    encode_dim: int = None

    registry = ModelRegistry()

    def __init_subclass__(
        cls,
        key=None,
        abstract=False,
    ):
        if abstract:
            return
        if key is None:
            raise ValueError(
                f"{cls.__name__} doesn't define `key` in class definition."
            )
        cls.key = key
        # Allow multiple names
        if isinstance(key, str):
            key = list([key])
        for k in key:
            if k in cls.registry:
                raise ValueError(f"Model name {key} already registered.")
            cls.registry[k] = cls
        if cls.task != ModelTask.cv_feature:
            cls._field_validate()

        old_doc = cls.__doc__
        if old_doc is None:
            old_doc = ""
        inject_doc = model_doc(cls)
        cls.__doc__ = old_doc + inject_doc

    def _repr_html_(self):
        return model_repr_html(self)

    def get_transform(self):
        return None

    def to(self, device):
        self.model.to(device)
        return self

    @staticmethod
    def load_weights(url, progress=True):
        from timm.models.hub import download_cached_file

        return Path(download_cached_file(url, progress=progress))

    def estimate_param_size(self):
        """Count the number of parameters in a model."""
        model = self.model
        if not isinstance(model, torch.nn.Module):
            try:
                # If it's a Coco model, get the underlying PyTorch model
                model = model.model
            except (AttributeError, TypeError):
                return None
        return sum(p.numel() for p in self.model.parameters())

    @classmethod
    def _field_validate(cls):
        attrs = {
            "task": cls.task,
            "description": cls.description,
            "license": cls.license,
            "commercial": cls.commercial,
        }
        missing_attrs = []
        for k, v in attrs.items():
            if v is None:
                missing_attrs.append(k)
        if len(missing_attrs) > 0:
            raise ValueError(
                f"Attributes {', '.join(missing_attrs)} is missing for {cls.__name__}."
            )

    @property
    def name(self):
        return self.__class__.__name__


class ImageModel(ModelBase, abstract=True):
    # TODO: Add a config that specify the recommended input tile size and mpp

    def get_transform(self):
        import torch
        from torchvision.transforms.v2 import (
            Compose,
            Normalize,
            Resize,
            ToDtype,
            ToImage,
        )

        return Compose(
            [
                ToImage(),
                ToDtype(dtype=torch.float32, scale=True),
                Resize(size=(224, 224), antialias=False),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    @abstractmethod
    def encode_image(self, image):
        raise NotImplementedError

    def __call__(self, image):
        return self.encode_image(image)


class TimmModel(ImageModel, abstract=True):
    def __init__(self, name, token=None, compile=False, compile_kws=None, **kwargs):
        import timm
        from huggingface_hub import login

        if token is not None:
            login(token)

        default_kws = {"pretrained": True, "num_classes": 0}
        default_kws.update(kwargs)

        with hf_access(name):
            self.model = timm.create_model(name, **default_kws)

        if compile:
            if compile_kws is None:
                compile_kws = {}
            self.compiled_model = torch.compile(self.model, **compile_kws)

    def get_transform(self):
        return get_default_transform()

    @torch.inference_mode()
    def encode_image(self, image):
        with torch.inference_mode():
            return self.model(image)


class SlideEncoderModel(ModelBase, abstract=True):
    @abstractmethod
    def encode_slide(self, embeddings, coords=None, **kwargs):
        raise NotImplementedError


class ImageTextModel(ImageModel, abstract=True):
    @abstractmethod
    def encode_image(self, image):
        """This should return the image feature before normalize."""
        raise NotImplementedError

    @abstractmethod
    def encode_text(self, text):
        raise NotImplementedError

    def tokenize(self, text):
        raise NotImplementedError


class SegmentationModel(ModelBase, abstract=True):
    probability_map_key = "probability_map"
    instance_map_key = "instance_map"
    class_map_key = "class_map"
    token_map_key = "token_map"

    def get_transform(self):
        import torch
        from torchvision.transforms.v2 import Compose, Normalize, ToDtype, ToImage

        return Compose(
            [
                ToImage(),
                ToDtype(dtype=torch.float32, scale=True),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    @abstractmethod
    def segment(self, image):
        raise NotImplementedError

    @abstractmethod
    def supported_output(self):
        return (
            "probability_map",
            "instance_map",
            "class_map",
            "token_map",
        )

    def get_classes(self):
        return None


class TilePredictionModel(ModelBase, abstract=True):
    @abstractmethod
    def predict(self, image):
        """The output should always be a dict of numpy arrays
        to allow multiple outputs.
        """
        raise NotImplementedError
