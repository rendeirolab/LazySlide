import torch

from .._utils import hf_access
from ..base import ImageModel


class Titan(ImageModel):
    name = "titan"

    TEMPLATES = [
        "CLASSNAME.",
        "an image of CLASSNAME.",
        "the image shows CLASSNAME.",
        "the image displays CLASSNAME.",
        "the image exhibits CLASSNAME.",
        "an example of CLASSNAME.",
        "CLASSNAME is shown.",
        "this is CLASSNAME.",
        "I observe CLASSNAME.",
        "the pathology image shows CLASSNAME.",
        "a pathology image shows CLASSNAME.",
        "the pathology slide shows CLASSNAME.",
        "shows CLASSNAME.",
        "contains CLASSNAME.",
        "presence of CLASSNAME.",
        "CLASSNAME is present.",
        "CLASSNAME is observed.",
        "the pathology image reveals CLASSNAME.",
        "a microscopic image of showing CLASSNAME.",
        "histology shows CLASSNAME.",
        "CLASSNAME can be seen.",
        "the tissue shows CLASSNAME.",
        "CLASSNAME is identified.",
    ]

    def __init__(self, model_path=None, token=None):
        from transformers import AutoModel

        with hf_access(model_path):
            self.model = AutoModel.from_pretrained(
                "MahmoodLab/TITAN",
                add_pooling_layer=False,
                use_auth_token=token,
                trust_remote_code=True,
            )
            self.conch, self.conch_transform = self.model.return_conch()

    def to(self, device):
        super().to(device)
        self.conch.to(device)

    def get_transform(self):
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.v2 import (
            Resize,
            CenterCrop,
            ToImage,
            ToDtype,
            Normalize,
            Compose,
        )

        return Compose(
            [
                ToImage(),
                Resize(448, interpolation=InterpolationMode.BICUBIC, antialias=True),
                CenterCrop(448),
                ToDtype(dtype=torch.float32, scale=True),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    @torch.inference_mode()
    def encode_image(self, image):
        image_feature = self.conch(image)
        return image_feature.detach().cpu().numpy()

    @torch.inference_mode()
    def encode_slide(self, embeddings, coords=None, base_tile_size=None, **kwargs):
        slide_embeddings = self.model.encode_slide_from_patch_features(
            embeddings, coords, base_tile_size
        )
        return slide_embeddings.detach().cpu().numpy()

    @torch.inference_mode()
    def score(
        self, slide_embeddings, prompts: list[str], template: str = None, **kwargs
    ):
        if template is None:
            template = self.TEMPLATES

        classifier = self.model.zero_shot_classifier(prompts, template)
        scores = self.model.zero_shot(slide_embeddings, classifier)
        return scores.squeeze(0).detach().cpu().numpy()
