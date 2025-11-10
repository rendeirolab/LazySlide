from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any

import torch

from ._repr import model_repr_html
from ._utils import get_default_transform, hf_access

if TYPE_CHECKING:
    from wsidata import TileSpec


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

    def _repr_html_(self):
        return model_repr_html(self)

    def get_transform(self):
        return None

    def to(self, device):
        self.model.to(device)
        return self

    def estimate_param_size(self) -> int | None:
        """Count the number of parameters in a model."""
        model = self.model
        if not isinstance(model, torch.nn.Module):
            try:
                # If it's a Coco model, get the underlying PyTorch model
                model = model.model
            except (AttributeError, TypeError):
                return None
        return sum(p.numel() for p in model.parameters())

    @property
    def name(self):
        return self.__class__.__name__

    @classmethod
    def check_input_tile(cls, tile_spec: "TileSpec") -> bool:
        """
        A helper function to check if the input tile size is valid.

        Return True if the input tile size is valid. And the model will be executed.
        Add a warning here if the input is not optimal but still can be executed.
        """
        return True


class ImageModel(ModelBase):
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


class TimmModel(ImageModel):
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


class SlideEncoderModel(ModelBase):
    @abstractmethod
    def encode_slide(self, embeddings, coords=None, **kwargs):
        raise NotImplementedError


class ImageTextModel(ImageModel):
    @abstractmethod
    def encode_image(self, image):
        """This should return the image feature before normalize."""
        raise NotImplementedError

    @abstractmethod
    def encode_text(self, text):
        raise NotImplementedError

    def tokenize(self, text):
        raise NotImplementedError


class SegmentationModel(ModelBase):
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
    def segment(self, image) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def supported_output(self):
        return (
            "probability_map",
            "instance_map",
            "class_map",
            "token_map",
        )

    @staticmethod
    def get_classes():
        return None


class TilePredictionModel(ModelBase):
    @abstractmethod
    def predict(self, image):
        """The output should always be a dict of numpy arrays
        to allow multiple outputs.
        """
        raise NotImplementedError


class StyleTransferModel(ModelBase):
    @abstractmethod
    def predict(self, image):
        raise NotImplementedError

    @abstractmethod
    def get_channel_names(self):
        raise NotImplementedError
