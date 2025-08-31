import warnings

import torch

from ..._utils import find_stack_level
from .._utils import hf_access
from ..base import ImageTextModel, ModelTask


class CONCH(ImageTextModel, key="conch"):
    is_gated = True
    task = ModelTask.multimodal
    license = "CC-BY-NC-ND-4.0"
    description = "CONtrastive learning from Captions for Histopathology (CONCH)"
    commercial = False
    hf_url = "https://huggingface.co/MahmoodLab/conch"
    github_url = "https://github.com/mahmoodlab/CONCH"
    paper_url = "https://doi.org/10.1038/s41591-024-02856-4"
    bib_key = "Lu2024-nu"
    param_size = "395.2M"
    encode_dim = 512

    def __init__(self, model_path=None, token=None):
        warnings.warn(
            "As from v0.8.2, Normalization will not be applied to image embedding of CONCH model anymore."
            "A `normalize=True` argument is added to the `text_image_similarity` method."
            "If you only use the image embedding for text image similarity, you can safely ignore this warning.",
            stacklevel=find_stack_level(),
        )
        try:
            from conch.open_clip_custom import (
                create_model_from_pretrained,
                get_tokenizer,
            )
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

        # Move image to the same device as the model
        # Get the model device
        try:
            device = next(self.model.parameters()).device
        except Exception:
            device = torch.device("cpu")
        image = image.to(device)

        image_feature = self.model.encode_image(
            image, normalize=False, proj_contrast=True
        )
        return image_feature

    def tokenize(self, text):
        from conch.open_clip_custom import tokenize

        return tokenize(self.tokenizer, text)

    @torch.inference_mode()
    def encode_text(self, text):
        encode_texts = self.tokenize(text)
        # Move tokenized text to the same device as the model
        # Get the model device
        try:
            device = next(self.model.parameters()).device
        except Exception:
            device = torch.device("cpu")
        encode_texts = encode_texts.to(device)
        text_feature = self.model.encode_text(encode_texts)
        return text_feature
