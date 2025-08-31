import torch

from lazyslide.models._utils import hf_access
from lazyslide.models.base import ModelTask, SlideEncoderModel


class MadeleineSlideEncoder(SlideEncoderModel, key="madeleine"):
    task = ModelTask.slide_encoder
    license = "CC BY-NC-ND 4.0"
    description = (
        "Multistain Pretraining for Slide Representation Learning in Pathology"
    )
    commercial = False
    hf_url = "https://huggingface.co/MahmoodLab/madeleine"
    github_url = "https://github.com/mahmoodlab/MADELEINE"
    paper_url = "http://arxiv.org/abs/2408.02859"
    bib_key = "Jaume2024-tq"
    param_size = "3.2M"

    def __init__(self, model_path=None, token=None):
        from huggingface_hub import hf_hub_download

        with hf_access("MahmoodLab/madeleine"):
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
