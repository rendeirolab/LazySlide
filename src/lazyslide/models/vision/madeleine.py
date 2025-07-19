import torch

from lazyslide.models.base import SlideEncoderModel


class MadeleineSlideEncoder(SlideEncoderModel):
    def __init__(self, model_path=None, token=None):
        from huggingface_hub import hf_hub_download

        model_file = hf_hub_download(
            "RendeiroLab/LazySlide-models", "MADELEINE/madeleine_jit.pt"
        )

        self.model = torch.jit.load(model_file, map_location="cpu")

    def encode_slide(self, embeddings, coords=None, **kwargs):
        """
        Encode the slide using the Madeleine slide encoder.
        The embeddings should be a tensor of shape [B, C, H, W].
        """
        with torch.inference_mode():
            if len(embeddings.shape) == 2:
                # If embeddings are of shape [T, N], we need to unsqueeze to [1, T, N]
                embeddings = embeddings.unsqueeze(0)
            output = self.model(embeddings)
            return output
