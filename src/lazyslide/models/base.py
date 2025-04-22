from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import timm
import torch
from huggingface_hub import login
from timm.models.hub import download_cached_file

from lazyslide.models._utils import hf_access, get_default_transform


class ModelBase:
    model: torch.nn.Module
    name: str = "ModelBase"
    is_restricted: bool = False

    def get_transform(self):
        return None

    def to(self, device):
        self.model.to(device)
        return self

    @staticmethod
    def load_weights(url, progress=True):
        return Path(download_cached_file(url, progress=progress))


class ImageModel(ModelBase):
    # TODO: Add a config that specify the recommended input tile size and mpp

    def get_transform(self):
        import torch
        from torchvision.transforms.v2 import (
            Compose,
            ToImage,
            ToDtype,
            Resize,
            Normalize,
        )

        return Compose(
            [
                ToImage(),
                ToDtype(dtype=torch.float32, scale=True),
                Resize(size=(224, 224), antialias=False),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def encode_image(self, image) -> np.ndarray[np.float32]:
        raise NotImplementedError

    def __call__(self, image):
        return self.encode_image(image)


class TimmModel(ImageModel):
    def __init__(self, name, token=None, compile=False, compile_kws=None, **kwargs):
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
            return self.model(image).cpu().detach().numpy()


class SlideEncoderModel(ModelBase):
    def encode_slide(self, embeddings, coords=None):
        raise NotImplementedError


class ImageTextModel(ImageModel):
    def encode_image(self, image):
        """This should return the image feature before normalize."""
        raise NotImplementedError

    def encode_text(self, text):
        raise NotImplementedError

    def tokenize(self, text):
        raise NotImplementedError


class SegmentationModel(ModelBase):
    CLASS_MAPPING = None

    def get_transform(self):
        import torch
        from torchvision.transforms.v2 import Compose, ToImage, ToDtype, Normalize

        return Compose(
            [
                ToImage(),
                ToDtype(dtype=torch.float32, scale=True),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def segment(self, image):
        raise NotImplementedError

    def get_postprocess(self) -> Callable | None:
        return None
