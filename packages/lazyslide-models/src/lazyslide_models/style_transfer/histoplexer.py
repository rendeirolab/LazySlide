import torch

from lazyslide.models.base import ModelTask, StyleTransferModel


class Histoplexer(StyleTransferModel, key="histoplexer"):
    task = ModelTask.style_transfer
    license = "CC-BY-NC-ND-4.0"
    description = "Histopathology-based Protein Multiplex Generation"
    commercial = False
    github_url = "https://github.com/ratschlab/HistoPlexer"
    paper_url = "https://doi.org/10.1038/s42256-025-01074-y"
    bib_key = "Andani2025-jq"
    param_size = "27.5M"

    def __init__(self, model_path=None, token=None):
        from huggingface_hub import hf_hub_download

        model_file = hf_hub_download(
            "RendeiroLab/LazySlide-models", "histoplexer/histoplexer_jit.pt"
        )
        self.model = torch.jit.load(model_file, map_location="cpu")

    def predict(self, image):
        return self.model(image)
