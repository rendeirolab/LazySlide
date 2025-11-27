import torch

from .._model_registry import register
from ..base import ModelTask, TimmModel


@register(
    key="path_orchestra",
    is_gated=True,
    task=ModelTask.vision,
    license="CC-BY-NC-ND-4.0",
    description="Foundation Model for Computational Pathology",
    commercial=False,
    hf_url="https://huggingface.co/AI4Pathology/PathOrchestra",
    github_url="https://github.com/yanfang-research/PathOrchestra",
    bib_key="Yan2025-nc",
    # param_size="1.13B", # TODO
    # encode_dim=1536, # TODO
)
class PathOrchestra(TimmModel):
    def __init__(self, model_path=None, token=None):
        super().__init__(
            "hf-hub:AI4Pathology/PathOrchestra",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=False,
            token=token,
        )

    def get_transform(self):
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
                Resize(224),
                ToDtype(torch.float32, scale=True),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
