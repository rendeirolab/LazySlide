import timm
import torch

from lazyslide.models.base import TimmModel


def get_uni_transform():
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
            Resize(
                size=(224, 224),
                antialias=True,
                interpolation=InterpolationMode.BILINEAR,
                max_size=None,
            ),
            CenterCrop(size=(224, 224)),
            ToDtype(dtype=torch.float32, scale=True),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


class UNI(TimmModel):
    def __init__(self, model_path=None, token=None):
        # from huggingface_hub import hf_hub_download
        # model_path = hf_hub_download("MahmoodLab/UNI", filename="pytorch_model.bin")

        if model_path is not None:
            super().__init__(
                "vit_large_patch16_224",
                token=token,
                img_size=224,
                patch_size=16,
                init_values=1e-5,
                num_classes=0,
                dynamic_img_size=True,
                pretrained=False,
            )
            self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        else:
            super().__init__(
                "hf-hub:MahmoodLab/uni",
                token=token,
                init_values=1e-5,
                dynamic_img_size=True,
            )

    def get_transform(self):
        return get_uni_transform()


class UNI2(TimmModel):
    def __init__(self, model_path=None, token=None):
        timm_kwargs = {
            "img_size": 224,
            "patch_size": 14,
            "depth": 24,
            "num_heads": 24,
            "init_values": 1e-5,
            "embed_dim": 1536,
            "mlp_ratio": 2.66667 * 2,
            "num_classes": 0,
            "no_embed_class": True,
            "mlp_layer": timm.layers.SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
            "reg_tokens": 8,
            "dynamic_img_size": True,
        }

        # from huggingface_hub import hf_hub_download
        # model_path = hf_hub_download("MahmoodLab/UNI2-h", filename="pytorch_model.bin")

        if model_path is not None:
            super().__init__(
                "vit_giant_patch14_224", token=token, pretrained=False, **timm_kwargs
            )
            self.model.load_state_dict(
                torch.load(model_path, map_location="cpu"), strict=True
            )
        else:
            super().__init__("hf-hub:MahmoodLab/UNI2-h", **timm_kwargs)

    def get_transform(self):
        return get_uni_transform()
