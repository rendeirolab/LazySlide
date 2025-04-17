import torch
from timm.layers import SwiGLUPacked

from lazyslide.models.base import TimmModel, ModelBase


def get_virchow_transform():
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.v2 import (
        Compose,
        Normalize,
        CenterCrop,
        ToImage,
        ToDtype,
        Resize,
    )

    transforms = [
        ToImage(),
        Resize(
            size=(224, 224),
            interpolation=InterpolationMode.BICUBIC,
            max_size=None,
            antialias=True,
        ),
        CenterCrop(224),
        ToDtype(dtype=torch.float32, scale=True),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
    return Compose(transforms)


class Virchow(TimmModel):
    _hf_hub_id = "paige-ai/Virchow"

    def __init__(self, model_path=None, token=None):
        super().__init__(
            f"hf-hub:{self._hf_hub_id}",
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
            token=token,
        )

    def get_transform(self):
        return get_virchow_transform()

    def encode_image(self, img):
        with torch.inference_mode():
            output = self.model(img)
            # CLS token features (1, 768):
            cls_features = output[:, 0]
            # Patch token features (1, 256, 768):
            patch_features = output[:, self.model.num_prefix_tokens :]
        return torch.cat((cls_features, patch_features.mean(1)), dim=-1)


class Virchow2(Virchow):
    _hf_hub_id = "paige-ai/Virchow2"
