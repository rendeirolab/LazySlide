import torch

from .._model_registry import register
from .._utils import hf_access
from ..base import ImageTextModel, ModelTask


@register(
    key="medsiglip",
    is_gated=True,
    task=ModelTask.multimodal,
    license="health-ai-developer-foundations",
    license_url="https://developers.google.com/health-ai-developer-foundations/terms",
    description="MedSigLip is a variant of SigLip from Google for medical image analysis.",
    commercial=False,
    hf_url="https://huggingface.co/google/medsiglip-448",
    github_url="https://github.com/google-health/medsiglip",
    paper_url="https://arxiv.org/abs/2507.05201",
    bib_key="Sellergren2025-qq",
    encode_dim=1152,
    param_size="878M",
)
class MedSigLip(ImageTextModel):
    def __init__(self, model_path=None, token=None):
        try:
            from transformers import AutoModel, AutoProcessor
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "To run MedSigLip model, 'transformers' must be installed, try "
                "`pip install transformers`."
            )

        with hf_access("google/medsiglip-448"):
            self.model = AutoModel.from_pretrained("google/medsiglip-448")
            self.processor = AutoProcessor.from_pretrained(
                "google/medsiglip-448", use_fast=True
            )
            self.model.eval()

    def get_transform(self):
        from torchvision.transforms.v2 import Compose, Resize, ToDtype, ToImage

        return Compose([ToImage(), Resize(448), ToDtype(torch.float32, scale=False)])

    @torch.inference_mode()
    def encode_image(self, image):
        inputs = self.processor(images=image, padding="max_length", return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        # Get image embeddings from the model output
        image_features = self.model.get_image_features(**inputs)
        return image_features

    @torch.inference_mode()
    def encode_text(self, text):
        inputs = self.processor(text=text, padding="max_length", return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        # Get text embeddings from the model output
        text_features = self.model.get_text_features(**inputs)
        # Normalize the features
        text_features = torch.nn.functional.normalize(text_features, p=2, dim=-1)
        return text_features
