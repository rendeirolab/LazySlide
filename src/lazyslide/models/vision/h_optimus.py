import torch

from lazyslide.models.base import ModelTask, TimmModel


def get_hoptimus_transform():
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
            Normalize(
                mean=(0.707223, 0.578729, 0.703617), std=(0.211883, 0.230117, 0.177517)
            ),
        ]
    )


class HOptimus0(TimmModel, key="h-optimus-0"):
    task = ModelTask.vision
    license = "Apache 2.0"
    description = "Vision foundation model"
    commercial = True
    hf_url = "https://huggingface.co/bioptimus/H-optimus-0"
    github_url = "https://github.com/bioptimus"
    bib_key = "Saillard2024-ho"
    param_size = "1.13B"
    encode_dim = 1536

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


class HOptimus1(TimmModel, key="h-optimus-1"):
    is_gated = True
    task = ModelTask.vision
    license = "CC-BY-NC-ND-4.0"
    description = "Vision foundation model"
    commercial = False
    hf_url = "https://huggingface.co/bioptimus/H-optimus-1"
    github_url = "https://github.com/bioptimus"
    bib_key = "Bioptimus2025-lj"
    param_size = "1.13B"
    encode_dim = 1536

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


class H0Mini(TimmModel, key="h0-mini"):
    is_gated = True
    task = ModelTask.vision
    license = "CC-BY-NC-ND-4.0"
    description = "A distilled version of H-optimus-0"
    commercial = False
    hf_url = "https://huggingface.co/bioptimus/H0-mini"
    github_url = "https://github.com/bioptimus"
    paper_url = "https://doi.org/10.48550/arXiv.2501.16239"
    bib_key = "Filiot2025-bn"
    param_size = "85.7M"
    encode_dim = 1536

    def __init__(self, model_path=None, token=None):
        import timm

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
        return concatenated_features
