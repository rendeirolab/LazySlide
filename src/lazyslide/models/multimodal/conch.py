import torch

from .._utils import hf_access
from ..base import ImageTextModel


class CONCH(ImageTextModel):
    def __init__(self, model_path=None, token=None):
        try:
            from conch.open_clip_custom import create_model_from_pretrained
            from conch.open_clip_custom import get_tokenizer
        except ImportError:
            raise ImportError(
                "Conch is not installed. You can install it using "
                "`pip install git+https://github.com/mahmoodlab/CONCH.git`."
            )

        if model_path is None:
            model_path = "hf_hub:MahmoodLab/conch"

        with hf_access(model_path):
            self.model, self.processor = create_model_from_pretrained(
                "conch_ViT-B-16", model_path, hf_auth_token=token
            )
            self.tokenizer = get_tokenizer()

    @torch.inference_mode()
    def encode_image(self, image):
        if not isinstance(image, torch.Tensor):
            image = self.processor(image)
        if image.dim() == 3:
            image = image.unsqueeze(0)

        image_feature = self.model.encode_image(
            image, normalize=True, proj_contrast=True
        )
        return image_feature

    def tokenize(self, text):
        from conch.open_clip_custom import tokenize

        return tokenize(self.tokenizer, text)

    @torch.inference_mode()
    def encode_text(self, text):
        encode_texts = self.tokenize(text)
        text_feature = self.model.encode_text(encode_texts)
        return text_feature
