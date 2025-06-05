import torch

from lazyslide.models._utils import hf_access
from lazyslide.models.base import ImageModel


class Hibou(ImageModel):
    def __init__(self, hibou_version: str, model_path=None, token=None):
        try:
            from transformers import AutoModel
        except ImportError:
            raise ImportError(
                "transformers is not installed. You can install it using "
                "`pip install transformers`."
            )

        self.version = hibou_version

        with hf_access(f"histai/{self.version}"):
            self.model = AutoModel.from_pretrained(
                f"histai/{self.version}", trust_remote_code=True
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


class HibouB(Hibou):
    def __init__(self, token=None, model_path=None):
        super().__init__(hibou_version="hibou-b", token=token)


class HibouL(Hibou):
    def __init__(self, token=None, model_path=None):
        super().__init__(hibou_version="hibou-l", token=token)
