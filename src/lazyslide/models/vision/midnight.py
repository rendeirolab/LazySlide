import torch

from lazyslide.models._utils import hf_access
from lazyslide.models.base import ImageModel


class Midnight(ImageModel):
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
