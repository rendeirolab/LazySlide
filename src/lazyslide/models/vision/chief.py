import torch

from lazyslide.models.base import ImageModel, SlideEncoderModel


class CHIEF(ImageModel):
    name = "chief"

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


class CHIEFSlideEncoder(SlideEncoderModel):
    def __init__(self, model_path=None, token=None):
        from huggingface_hub import hf_hub_download

        model_file = hf_hub_download(
            "RendeiroLab/LazySlide-models-gpl", "CHIEF/CHIEF_slide_encoder_jit.pt"
        )

        self.model = torch.jit.load(model_file, map_location="cpu")

    def encode_slide(self, embeddings, coords=None):
        """
        Encode the slide using the CHIEF slide encoder.
        The embeddings should be a tensor of shape [B, N].
        """

        if len(embeddings.shape) == 3:
            raise ValueError(
                "CHIEF slide encoder expects a 2D tensor of shape [B, N], "
                "B tiles with N features."
            )

        with torch.inference_mode():
            output = self.model(embeddings)
            return output
