import torch

from lazyslide.models._utils import hf_access
from lazyslide.models.base import ImageModel


class PLIPVision(ImageModel):
    def __init__(self, model_path=None, token=None):
        try:
            from transformers import CLIPVisionModelWithProjection, CLIPProcessor
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
                model_path, use_auth_token=token
            )

    def get_transform(self):
        return None

    @torch.inference_mode()
    def encode_image(self, image):
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        image_features = self.model.get_image_features(**inputs)
        return image_features
