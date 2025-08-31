import torch

from lazyslide.models._utils import hf_access
from lazyslide.models.base import ImageModel, ModelTask


class PLIPVision(ImageModel, key="plip_vision"):
    task = ModelTask.vision
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
        raise Exception(
            "plip_vision is deprecated and will be removed in v0.10.0. "
            "Use plip directly instead."
        )
        try:
            from transformers import CLIPProcessor, CLIPVisionModelWithProjection
        except ImportError:
            raise ImportError(
                "Please install the 'transformers' package to use the PLIP model"
            )

        super().__init__()

        if model_path is None:
            model_path = "vinid/plip"

        with hf_access(model_path):
            self.model = CLIPVisionModelWithProjection.from_pretrained(
                model_path, use_auth_token=token
            )
            self.processor = CLIPProcessor.from_pretrained(
                model_path, use_auth_token=token, use_fast=True, do_rescale=False
            )

    def get_transform(self):
        return None

    @torch.inference_mode()
    def encode_image(self, image):
        inputs = self.processor(
            images=image,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        image_features = self.model(**inputs)
        return image_features["image_embeds"]
