import torch

from lazyslide.models.base import ImageModel


class CTransPath(ImageModel):
    name = "ctranspath"

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
