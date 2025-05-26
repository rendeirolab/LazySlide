import torch

from lazyslide.models._utils import hf_access
from lazyslide.models.base import ImageModel


class Hibou(ImageModel):
    def __init__(self, hibou_version: str, token=None):
        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError:
            raise ImportError(
                "transformers is not installed. You can install it using "
                "`pip install transformers`."
            )

        self.version = hibou_version
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with hf_access(f"histai/{self.version}"):
            self.model = AutoModel.from_pretrained(
                f"histai/{self.version}", hf_auth_token=token, trust_remote_code=True
            ).to(self.device)
            self.processor = AutoImageProcessor.from_pretrained(
                f"histai/{self.version}", hf_auth_token=token, trust_remote_code=True
            )

    def encode_image(self, image):
        image = self.processor(images=image, return_tensors="pt").to(self.device)
        self.model.eval()
        with torch.no_grad():
            image_features = self.model(**image).pooler_output
        return image_features


class HibouB(Hibou):
    def __init__(self, token=None):
        super().__init__(hibou_version="hibou-b", token=token)


class HibouL(Hibou):
    def __init__(self, token=None):
        super().__init__(hibou_version="hibou-l", token=token)
