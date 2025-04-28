from functools import partial

import numpy as np
import torch

from lazyslide.models._utils import hf_access
from lazyslide.models.base import ImageModel


class Phikon(ImageModel):
    name = "phikon"

    def __init__(self, model_path=None, token=None):
        from transformers import AutoImageProcessor, ViTModel

        with hf_access("owkin/phikon"):
            self.model = ViTModel.from_pretrained(
                "owkin/phikon",
                add_pooling_layer=False,
                use_auth_token=token,
            )
            self.img_processor = AutoImageProcessor.from_pretrained(
                "owkin/phikon", use_fast=True
            )

    def get_transform(self):
        return None

    @torch.inference_mode()
    def encode_image(self, image) -> np.ndarray[np.float32]:
        inputs = self.img_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        return self.model(**inputs).last_hidden_state[:, 0, :].cpu().detach().numpy()


class PhikonV2(ImageModel):
    name = "phikon-v2"

    def __init__(self, model_path=None, token=None):
        from transformers import AutoImageProcessor, AutoModel

        with hf_access("owkin/phikon-v2"):
            self.model = AutoModel.from_pretrained(
                "owkin/phikon-v2",
                add_pooling_layer=False,
                use_auth_token=token,
            )
            self.img_processor = AutoImageProcessor.from_pretrained(
                "owkin/phikon-v2", use_fast=True
            )

    def get_transform(self):
        return None

    @torch.inference_mode()
    def encode_image(self, image) -> np.ndarray[np.float32]:
        inputs = self.img_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        return self.model(**inputs).last_hidden_state[:, 0, :].cpu().detach().numpy()
