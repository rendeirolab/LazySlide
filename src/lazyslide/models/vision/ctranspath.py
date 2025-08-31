import torch

from lazyslide.models.base import ImageModel, ModelTask


class CTransPath(ImageModel, key="ctranspath"):
    task = ModelTask.vision
    license = "GPL-3.0"
    description = "Transformer-based unsupervised contrastive learning for histopathological image classification"
    commercial = False
    github_url = "https://github.com/Xiyue-Wang/TransPath"
    paper_url = "https://doi.org/10.1016/j.media.2022.102559"
    bib_key = "Wang2022-rk"
    param_size = "27.5M"
    encode_dim = 768

    def __init__(self, model_path=None, token=None):
        from huggingface_hub import hf_hub_download

        model_file = hf_hub_download(
            "RendeiroLab/LazySlide-models-gpl", "CTransPath/ctranspath_jit.pt"
        )

        self.model = torch.jit.load(model_file, map_location="cpu")

    def encode_image(self, image):
        """
        Encode the input image using the CTransPath model.
        The model expects a tensor of shape [B, C, H, W].
        """
        with torch.inference_mode():
            output = self.model(image)
            return output
