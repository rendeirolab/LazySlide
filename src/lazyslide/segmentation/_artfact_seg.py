from typing import Literal

import torch
from wsidata import WSIData

from lazyslide._const import Key
from lazyslide.models.segmentation import SMPBase


class GrandQCArtifactSegmentation(SMPBase):
    def __init__(self, model: Literal["5x", "7x", "10x"] = "7x"):
        weights_map = {
            "5x": "GrandQC_MPP2.pth",
            "7x": "GrandQC_MPP10.pth",
            "10x": "GrandQC_MPP1.pth",
        }
        weights = self.load_weights(
            f"https://zenodo.org/records/14041538/files/{weights_map[model]}"
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
            torch.load(weights, map_location=torch.device("cpu"))
        )
        self.model.eval()

    def segment(self, image):
        with torch.inference_mode():
            return self.model.predict(image)


def artifacts_segmentation(
    wsi: WSIData,
    tissue_key: str = Key.tissue,
    key_added: str = "artifacts",
):
    pass


def offset_artifacts(
    wsi: WSIData,
    artifacts_key: str = "artifacts",
    key_added: str = "offset_artifacts",
):
    pass
