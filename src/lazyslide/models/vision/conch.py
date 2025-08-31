import torch

from lazyslide.models._utils import hf_access
from lazyslide.models.base import ImageModel, ModelTask


class CONCHVision(ImageModel, key="conch_vision"):
    is_gated = True
    task = ModelTask.vision
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
        raise Exception(
            "conch_vision is deprecated and will be removed in v0.10.0. "
            "Use conch directly instead."
        )
        try:
            from conch.open_clip_custom import create_model_from_pretrained
        except ImportError:
            raise ImportError(
                "Conch is not installed. You can install it using "
                "`pip install git+https://github.com/mahmoodlab/CONCH.git`."
            )

        if model_path is None:
            model_path = "hf_hub:MahmoodLab/conch"

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
