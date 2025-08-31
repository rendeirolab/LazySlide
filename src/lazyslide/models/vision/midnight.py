import torch

from lazyslide.models._utils import hf_access
from lazyslide.models.base import ImageModel, ModelTask


class Midnight(ImageModel, key="midnight"):
    is_gated = True
    task = ModelTask.vision
    license = "MIT"
    description = "Training state-of-the-art pathology foundation models with orders of magnitude less data"
    commercial = True
    hf_url = "https://huggingface.co/kaiko-ai/midnight"
    github_url = "https://github.com/kaiko-ai/midnight"
    paper_url = "https://doi.org/10.48550/arXiv.2504.05186"
    bib_key = "Karasikov2025-wp"
    param_size = "1.14B"
    encode_dim = 3072

    def __init__(self, model_path=None, token=None):
        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError:
            raise ImportError(
                "transformers is not installed. You can install it using "
                "`pip install transformers`."
            )

        with hf_access("kaiko-ai/midnight"):
            self.model = AutoModel.from_pretrained("kaiko-ai/midnight")

    def get_transform(self):
        from torchvision.transforms import v2

        return v2.Compose(
            [
                v2.ToImage(),
                v2.Resize(224),
                v2.CenterCrop(224),
                v2.ToDtype(dtype=torch.float32, scale=True),
                v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

    @staticmethod
    def extract_classification_embedding(tensor):
        cls_embedding = tensor[:, 0, :]
        patch_embedding = tensor[:, 1:, :].mean(dim=1)
        return torch.cat([cls_embedding, patch_embedding], dim=-1)

    @torch.inference_mode()
    def encode_image(self, image):
        output = self.model(image).last_hidden_state
        image_feature = self.extract_classification_embedding(output)
        return image_feature
