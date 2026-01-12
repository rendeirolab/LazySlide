import torch

from .._model_registry import register
from ..base import ModelTask, SegmentationModel


@register(
    key="hest-tissue-segmentation",
    task=ModelTask.segmentation,
    license="CC-BY-NC-SA-4.0",
    description="DeepLabV3 model finetuned on HEST-1k and Acrobat for IHC/H&E tissue segmentation.",
    commercial=False,
    hf_url="https://huggingface.co/MahmoodLab/hest-tissue-seg",
    param_size="39.6M",
    flops="62.61G",
)
class HESTTissueSegmentation(SegmentationModel):
    """
    Tissue segmentation model from HEST.

    512x512 with mpp=1 or 2
    """

    def __init__(self, model_path=None, token=None):
        from huggingface_hub import hf_hub_download

        model_file = hf_hub_download(
            "RendeiroLab/LazySlide-models",
            "HEST/hest_tissue_seg_jit.pt",
        )

        self.model = torch.jit.load(model_file, map_location="cpu")
        self.model.eval()

    def get_transform(self):
        from torchvision.transforms.v2 import (
            Compose,
            Normalize,
            ToDtype,
            ToImage,
        )

        return Compose(
            [
                ToImage(),
                ToDtype(dtype=torch.float32, scale=True),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def segment(self, image):
        with torch.inference_mode():
            return {"probability_map": self.model(image)["out"].softmax(1)}

    def supported_output(self):
        return ("probability_map",)
