import timm
import torch
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from lazyslide.models.base import TimmModel


def get_hoptimus_transform():
    from torchvision.transforms.v2 import (
        Compose,
        ToImage,
        Resize,
        CenterCrop,
        ToDtype,
        Normalize,
    )
    from torchvision.transforms import InterpolationMode

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
            Normalize(
                mean=(0.707223, 0.578729, 0.703617), std=(0.211883, 0.230117, 0.177517)
            ),
        ]
    )


class HOptimus0(TimmModel):
    name = "H-optimus-0"

    def __init__(self, model_path=None, token=None):
        super().__init__(
            "hf-hub:bioptimus/H-optimus-0",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=False,
            token=token,
        )

    def get_transform(self):
        return get_hoptimus_transform()


class HOptimus1(TimmModel):
    name = "H-optimus-1"

    def __init__(self, model_path=None, token=None):
        super().__init__(
            "hf-hub:bioptimus/H-optimus-1",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=False,
            token=token,
        )

    def get_transform(self):
        return get_hoptimus_transform()


class H0Mini(TimmModel):
    name = "H0-mini"

    def __init__(self, model_path=None, token=None):
        super().__init__(
            "hf-hub:bioptimus/H0-mini",
            pretrained=True,
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=torch.nn.SiLU,
            token=token,
        )

    def get_transform(self):
        return get_hoptimus_transform()

    @torch.inference_mode()
    def encode_image(self, image):
        output = self.model(image)
        # CLS token features (1, 768):
        cls_features = output[:, 0]
        # Patch token features (1, 256, 768):
        patch_token_features = output[:, self.model.num_prefix_tokens :]
        # Concatenate the CLS token features with the mean of the patch token
        # features (1, 1536):
        concatenated_features = torch.cat(
            [cls_features, patch_token_features.mean(1)], dim=-1
        )
        return concatenated_features.cpu().detach().numpy()
