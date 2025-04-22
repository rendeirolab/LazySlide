# Modified from https://github.com/PathologyFoundation/plip/blob/main/plip.py

import torch

from .._utils import hf_access
from ..base import ImageTextModel


class PLIP(ImageTextModel):
    def __init__(self, model_path=None, token=None):
        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError:
            raise ImportError(
                "Please install the 'transformers' package to use the PLIP model"
            )

        if model_path is None:
            model_path = "vinid/plip"

        with hf_access(model_path):
            self.model = CLIPModel.from_pretrained(model_path, use_auth_token=token)
            self.processor = CLIPProcessor.from_pretrained(
                model_path, use_auth_token=token
            )

    def get_transform(self):
        return None

    @torch.inference_mode()
    def encode_image(self, image):
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        image_features = self.model.get_image_features(**inputs)
        image_features = torch.nn.functional.normalize(image_features, p=2, dim=-1)
        return image_features

    @torch.inference_mode()
    def encode_text(self, text):
        inputs = self.processor(
            text=text,
            return_tensors="pt",
            max_length=77,
            padding="max_length",
            truncation=True,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        text_features = self.model.get_text_features(**inputs)
        return text_features
