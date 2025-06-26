import torch

from lazyslide.models.base import SlideEncoderModel


class MadeleineSlideEncoder(SlideEncoderModel):
    def __init__(self, model_path=None, token=None):
        from huggingface_hub import hf_hub_download

        model_file = hf_hub_download(
            "RendeiroLab/LazySlide-models", "MADELEINE/madeleine_jit.pt"
        )

        self.model = torch.jit.load(model_file, map_location="cpu")

    def encode_slide(self, embeddings, coords=None):
        """
        Encode the slide using the Madeleine slide encoder.
        The embeddings should be a tensor of shape [B, C, H, W].
        """
        with torch.inference_mode():
            output = self.model(embeddings)
            return output
