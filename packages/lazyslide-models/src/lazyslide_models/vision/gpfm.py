import torch
from huggingface_hub import hf_hub_download

from lazyslide.models.base import ImageModel, ModelTask

from .._model_registry import register


@register(
    key="gpfm",
    is_gated=False,
    task=ModelTask.vision,
    license="CC BY-NC-ND 4.0",
    description="Generalizable Pathology Foundation Model",
    commercial=False,
    hf_url="https://huggingface.co/majiabo/GPFM",
    github_url="https://github.com/birkhoffkiki/GPFM",
    paper_url="https://doi.org/10.1038/s41551-025-01488-4",
    bib_key="Ma2025-wm",
    param_size="303M",
    encode_dim=1024,
    flops="155.53G",
)
class GPFM(ImageModel):
    def __init__(self, model_path=None, token=None):
        import timm

        ckpt_path = hf_hub_download(repo_id="majiabo/GPFM", filename="GPFM.pth")
        model = timm.create_model(
            "vit_large_patch14_dinov2.lvd142m",
            pretrained=False,
            img_size=224,
            init_values=1.0e-05,
        )
        state_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        self.model = model

    def get_transform(self):
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.v2 import (
            Compose,
            Normalize,
            Resize,
            ToDtype,
            ToImage,
        )

        return Compose(
            [
                ToImage(),
                Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                ToDtype(dtype=torch.float32, scale=True),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    @torch.inference_mode()
    def encode_image(self, image):
        return self.model(image)
