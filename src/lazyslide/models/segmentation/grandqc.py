from typing import Literal

import torch
from huggingface_hub import hf_hub_download
from lazyslide.models.base import SegmentationModel
from lazyslide.models.segmentation.postprocess import semanticseg_postprocess
from lazyslide.models.segmentation.smp import SMPBase


class GrandQCArtifact(SegmentationModel):
    CLASS_MAPPING = {
        0: "Background",
        1: "Normal Tissue",
        2: "Fold",
        3: "Darkspot & Foreign Object",
        4: "PenMarking",
        5: "Edge & Air Bubble",
        6: "Out of Focus",
        7: "Background",
    }

    def __init__(self, model: Literal["5x", "7x", "10x"] = "7x"):
        weights_map = {
            "5x": "GrandQC_MPP2_traced.pt",
            "7x": "GrandQC_MPP15_traced.pt",
            "10x": "GrandQC_MPP1_traced.pt",
        }
        weights = hf_hub_download(
            "RendeiroLab/LazySlide-models", f"grandqc/{weights_map[model]}"
        )

        self.model = torch.jit.load(weights)

    def get_transform(self):
        import torch
        from torchvision.transforms.v2 import (
            Compose,
            ToImage,
            ToDtype,
            Normalize,
        )

        return Compose(
            [
                ToImage(),
                ToDtype(dtype=torch.float32, scale=True),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    @torch.inference_mode()
    def segment(self, image):
        out = self.model(image)
        return out.detach().cpu().numpy()

    def get_postprocess(self):
        return semanticseg_postprocess


class GrandQCTissue(SMPBase):
    CLASS_MAPPING = {
        0: "Background",
        1: "Tissue",
    }

    def __init__(self):
        weights = hf_hub_download(
            "RendeiroLab/LazySlide-models", "grandqc/Tissue_Detection_MPP10.pth"
        )

        super().__init__(
            arch="unetplusplus",
            encoder_name="timm-efficientnet-b0",
            encoder_weights="imagenet",
            in_channels=3,
            classes=2,
            activation=None,
        )
        self.model.load_state_dict(
            torch.load(weights, map_location=torch.device("cpu"), weights_only=True)
        )
        self.model.eval()

    @torch.inference_mode()
    def segment(self, image):
        return self.model.predict(image)
