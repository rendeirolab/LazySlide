import torch

from lazyslide.models.base import ImageModel, ModelTask, SlideEncoderModel


class CHIEF(ImageModel, key="chief"):
    task = ModelTask.vision
    license = "AGPL-3.0"
    description = "Clinical Histopathology Imaging Evaluation Foundation (CHIEF)"
    commercial = False
    github_url = "https://github.com/hms-dbmi/CHIEF"
    paper_url = "https://doi.org/10.1038/s41586-024-07894-z"
    bib_key = "Wang2024-jb"
    param_size = "27.5M"
    encode_dim = 768

    def __init__(self, model_path=None, token=None):
        from huggingface_hub import hf_hub_download

        model_file = hf_hub_download(
            "RendeiroLab/LazySlide-models-gpl", "CHIEF/CHIEF_patch_encoder_jit.pt"
        )

        self.model = torch.jit.load(model_file, map_location="cpu")

    def encode_image(self, image):
        """
        Encode the input image using the CHIEF model.
        The model expects a tensor of shape [B, C, H, W].
        """
        with torch.inference_mode():
            output = self.model(image)
            return output


class CHIEFSlideEncoder(SlideEncoderModel, key="chief-slide-encoder"):
    task = ModelTask.slide_encoder
    license = "AGPL-3.0"
    description = "Clinical Histopathology Imaging Evaluation Foundation (CHIEF)"
    commercial = False
    github_url = "https://github.com/hms-dbmi/CHIEF"
    paper_url = "https://doi.org/10.1038/s41586-024-07894-z"
    bib_key = "Wang2024-jb"
    param_size = "1.2M"

    def __init__(self, model_path=None, token=None):
        from huggingface_hub import hf_hub_download

        model_file = hf_hub_download(
            "RendeiroLab/LazySlide-models-gpl", "CHIEF/CHIEF_slide_encoder_jit.pt"
        )

        self.model = torch.jit.load(model_file, map_location="cpu")

    def encode_slide(self, embeddings, coords=None, **kwargs):
        """
        Encode the slide using the CHIEF slide encoder.
        The embeddings should be a tensor of shape [B, T, N].
        T is the number of tiles, and N is the feature dimension.
        """

        with torch.inference_mode():
            if len(embeddings.shape) == 2:
                # If embeddings are of shape [T, N], we need to unsqueeze to [1, T, N]
                embeddings = embeddings.unsqueeze(0)
            outputs = []
            for emb in embeddings:
                output = self.model(emb)
                outputs.append(output.squeeze(0))
            return torch.stack(outputs, dim=0)
