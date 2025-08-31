import torch

from lazyslide.models._utils import hf_access
from lazyslide.models.base import ImageModel, ModelTask


class Hibou(ImageModel, abstract=True):
    is_gated = True
    task = ModelTask.vision
    license = "Apache 2.0"
    description = "A family of foundational vision transformers for pathology"
    commercial = True
    hf_url = "https://huggingface.co/histai/hibou-b"
    github_url = "https://github.com/HistAI/hibou/"
    paper_url = "https://doi.org/10.48550/arXiv.2406.05074"
    bib_key = "Nechaev2024-wi"

    def __init__(self, variant: str, model_path=None, token=None):
        try:
            from transformers import AutoModel
        except ImportError:
            raise ImportError(
                "transformers is not installed. You can install it using "
                "`pip install transformers`."
            )

        self.variant = variant

        with hf_access(f"histai/{self.variant}"):
            self.model = AutoModel.from_pretrained(
                f"histai/{self.variant}", trust_remote_code=True
            )

    def get_transform(self):
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.v2 import (
            CenterCrop,
            Compose,
            Normalize,
            Resize,
            ToDtype,
            ToImage,
        )

        return Compose(
            [
                ToImage(),
                Resize(
                    size=(224, 224),
                    interpolation=InterpolationMode.BICUBIC,
                    max_size=None,
                    antialias=True,
                ),
                CenterCrop(224),
                ToDtype(dtype=torch.float32, scale=True),
                Normalize(mean=(0.7068, 0.5755, 0.722), std=(0.195, 0.2316, 0.1816)),
            ]
        )

    @torch.inference_mode()
    def encode_image(self, image):
        image_features = self.model(pixel_values=image)
        return image_features.pooler_output


class HibouB(Hibou, key="hibou-b"):
    param_size = "85.7M"
    encode_dim = 768

    def __init__(self, token=None, model_path=None):
        super().__init__(variant="hibou-b", token=token)


class HibouL(Hibou, key="hibou-l"):
    param_size = "303.7M"
    encode_dim = 1024

    def __init__(self, token=None, model_path=None):
        super().__init__(variant="hibou-l", token=token)
