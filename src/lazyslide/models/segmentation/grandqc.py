from typing import Literal

import torch

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
        from huggingface_hub import hf_hub_download

        weights_map = {
            "5x": "GrandQC_MPP2_jit.pt",
            "7x": "GrandQC_MPP15_jit.pt",
            "10x": "GrandQC_MPP1_jit.pt",
        }
        weights = hf_hub_download(
            "RendeiroLab/LazySlide-models", f"GrandQC/{weights_map[model]}"
        )

        self.model = torch.jit.load(weights)

    def get_transform(self):
        import torch
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

    # @torch.inference_mode()
    def segment(self, image):
        with torch.inference_mode():
            out = self.model(image)
        return {"probability_map": out}

    def supported_output(self):
        return ("probability_map",)


# class GrandQCTissue(SMPBase):
#     CLASS_MAPPING = {
#         0: "Background",
#         1: "Tissue",
#     }
#
#     def __init__(self):
#         from huggingface_hub import hf_hub_download
#
#         weights = hf_hub_download(
#             "RendeiroLab/LazySlide-models", "grandqc/Tissue_Detection_MPP10.pth"
#         )
#
#         super().__init__(
#             arch="unetplusplus",
#             encoder_name="timm-efficientnet-b0",
#             encoder_weights="imagenet",
#             in_channels=3,
#             classes=2,
#             activation=None,
#         )
#         self.model.load_state_dict(
#             torch.load(weights, map_location=torch.device("cpu"), weights_only=True)
#         )
#         self.model.eval()
#
#     @torch.inference_mode()
#     def segment(self, image):
#         return {"probability_map": self.model.predict(image).softmax(dim=1)}
#
#     def supported_output(self):
#         return ("probability_map",)


class GrandQCTissue(SegmentationModel):
    CLASS_MAPPING = {
        0: "Background",
        1: "Tissue",
    }

    def __init__(self):
        from huggingface_hub import hf_hub_download

        weights = hf_hub_download(
            "RendeiroLab/LazySlide-models", "GrandQC/GrandQC_tissue_seg_jit.pt"
        )

        self.model = torch.jit.load(weights)
        self.model.eval()

    @torch.inference_mode()
    def segment(self, image):
        return {"probability_map": self.model(image).softmax(dim=1)}

    def supported_output(self):
        return ("probability_map",)
