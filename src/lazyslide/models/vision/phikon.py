import numpy as np
import torch

from lazyslide.models._utils import hf_access
from lazyslide.models.base import ImageModel, ModelTask


class Phikon(ImageModel, key="phikon"):
    task = ModelTask.vision
    license = "Owkin non-commercial license"
    license_url = "https://github.com/owkin/HistoSSLscaling/blob/main/LICENSE.txt"
    description = (
        "Scaling self-Supervised Learning for histopathology with Masked Image Modeling"
    )
    commercial = False
    hf_url = "https://huggingface.co/owkin/phikon"
    github_url = "https://github.com/owkin/HistoSSLscaling/"
    paper_url = "https://doi.org/10.1101/2023.07.21.23292757"
    bib_key = "Filiot2023-vg"
    param_size = "85.8M"
    encode_dim = 768

    def __init__(self, model_path=None, token=None):
        from transformers import AutoImageProcessor, ViTModel

        with hf_access("owkin/phikon"):
            self.model = ViTModel.from_pretrained(
                "owkin/phikon",
                add_pooling_layer=False,
                use_auth_token=token,
            )
            self.img_processor = AutoImageProcessor.from_pretrained(
                "owkin/phikon", use_fast=True
            )

    def get_transform(self):
        return None

    @torch.inference_mode()
    def encode_image(self, image) -> np.ndarray[np.float32]:
        inputs = self.img_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        return self.model(**inputs).last_hidden_state[:, 0, :]


class PhikonV2(ImageModel, key="phikonv2"):
    task = ModelTask.vision
    commercial = False
    license = "Owkin non-commercial license"
    license_url = "https://huggingface.co/owkin/phikon-v2/blob/main/LICENSE.pdf"
    description = "A large and public feature extractor for biomarker prediction"
    hf_url = "https://huggingface.co/owkin/phikon-v2"
    github_url = "https://github.com/owkin"
    paper_url = "https://doi.org/10.48550/arXiv.2409.09173"
    bib_key = "Filiot2024-at"
    param_size = "303.4M"
    encode_dim = 1024

    def __init__(self, model_path=None, token=None):
        from transformers import AutoImageProcessor, AutoModel

        with hf_access("owkin/phikon-v2"):
            self.model = AutoModel.from_pretrained(
                "owkin/phikon-v2",
                use_auth_token=token,
            )
            self.img_processor = AutoImageProcessor.from_pretrained(
                "owkin/phikon-v2", use_fast=True
            )

    def get_transform(self):
        return None

    @torch.inference_mode()
    def encode_image(self, image) -> np.ndarray[np.float32]:
        inputs = self.img_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        return self.model(**inputs).last_hidden_state[:, 0, :]
