import torch

from lazyslide.models._utils import hf_access
from lazyslide.models.base import ImageModel


class CONCHVision(ImageModel):
    def __init__(self, model_path=None, token=None):
        try:
            from conch.open_clip_custom import create_model_from_pretrained
        except ImportError:
            raise ImportError(
                "Conch is not installed. You can install it using "
                "`pip install git+https://github.com/mahmoodlab/CONCH.git`."
            )

        with hf_access("conch_ViT-B-16"):
            self.model, self.processor = create_model_from_pretrained(
                "conch_ViT-B-16", model_path, hf_auth_token=token
            )

    def get_transform(self):
        return None

    @torch.inference_mode()
    def encode_image(self, image):
        if not isinstance(image, torch.Tensor):
            image = self.processor(image)
        if image.dim() == 3:
            image = image.unsqueeze(0)

        image_feature = self.model.encode_image(
            image, normalize=False, proj_contrast=False
        )
        return image_feature
