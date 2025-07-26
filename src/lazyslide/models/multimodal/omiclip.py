import torch
import torch.nn.functional as F

from .._utils import hf_access
from ..base import ImageTextModel


class OmiCLIP(ImageTextModel):
    def __init__(self, model_path=None, token=None):
        try:
            from huggingface_hub import hf_hub_download
            from open_clip import create_model_from_pretrained, get_tokenizer
        except ImportError:
            raise ImportError(
                "open_clip is not installed. You can install it using "
                "`pip install open_clip_torch`."
            )

        if model_path is None:
            with hf_access("WangGuangyuLab/Loki"):
                model_path = hf_hub_download(
                    "WangGuangyuLab/Loki", "checkpoint.pt", token=token
                )

        self.model, self.preprocess = create_model_from_pretrained(
            "coca_ViT-L-14", pretrained=model_path, load_weights_only=False
        )
        self.tokenizer = get_tokenizer("coca_ViT-L-14")

    @torch.inference_mode()
    def encode_image(
        self,
        image,
    ) -> torch.Tensor:
        """
        Batch–encode a list of image file paths into L2‑normalized embeddings.
        Returns a tensor of shape (N, D).
        """

        if not isinstance(image, torch.Tensor):
            # Preprocess the image, then stack to create a batch of size 1
            image = self.processor(image)

        # Move image to the same device as the model
        try:
            device = next(self.model.parameters()).device
        except Exception:
            device = torch.device("cpu")
        image = image.to(device)

        # Generate the image features
        images_embedding = self.model.encode_image(image)

        # Normalize all embeddings across the feature dimension (L2 normalization)
        image_embeddings = F.normalize(images_embedding, p=2, dim=-1)

        return image_embeddings

    @torch.inference_mode()
    def encode_text(self, text):
        """
        Batch–encode a list of strings into L2‑normalized embeddings.
        Returns a tensor of shape (N, D).
        """
        # Tokenizer returns a dict of tensors
        text_inputs = self.tokenizer(text)

        # Move tokenized text to the same device as the model
        try:
            device = next(self.model.parameters()).device
        except Exception:
            device = torch.device("cpu")
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

        feats = self.model.encode_text(text_inputs)  # (N, D)
        normalized_features = F.normalize(feats, p=2, dim=-1)  # (N, D)
        return normalized_features
