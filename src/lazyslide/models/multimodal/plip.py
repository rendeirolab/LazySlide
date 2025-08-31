# Modified from https://github.com/PathologyFoundation/plip/blob/main/plip.py
import warnings

import torch

from ..._utils import find_stack_level
from .._utils import hf_access
from ..base import ImageTextModel, ModelTask


class PLIP(ImageTextModel, key="plip"):
    task = ModelTask.multimodal
    license = "Non-commercial"
    description = "Pathology Language-Image Pretraining (PLIP)"
    commercial = False
    hf_url = "https://huggingface.co/vinid/plip"
    github_url = "https://github.com/PathologyFoundation/plip"
    paper_url = "https://doi.org/10.1038/s41591-023-02504-3"
    bib_key = "Huang2023-wi"
    param_size = "87.8M"
    encode_dim = 512

    def __init__(self, model_path=None, token=None):
        warnings.warn(
            "As from v0.8.2, Normalization will not be applied to image embedding of PLIP model anymore."
            "A `normalize=True` argument is added to the `text_image_similarity` method."
            "If you only use the image embedding for text image similarity, you can safely ignore this warning.",
            stacklevel=find_stack_level(),
        )
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
        # image_features = torch.nn.functional.normalize(image_features, p=2, dim=-1)
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
