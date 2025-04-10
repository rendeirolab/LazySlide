from __future__ import annotations

from typing import Literal

import torch
from huggingface_hub import hf_hub_download
from wsidata import WSIData
from wsidata.io import add_shapes

from lazyslide._const import Key
from lazyslide._utils import get_torch_device
from lazyslide.models.base import SegmentationModel
from lazyslide.models.segmentation.postprocess import semanticseg_postprocess
from ._seg_runner import SegmentationRunner


class GrandQCArtifactSegmentation(SegmentationModel):
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

    def segment(self, image):
        with torch.inference_mode():
            out = self.model(image)
        return out.detach().cpu().numpy()

    def get_postprocess(self):
        return semanticseg_postprocess


# Define class mapping
CLASS_MAPPING = {
    1: "Normal Tissue",
    2: "Fold",
    3: "Darkspot & Foreign Object",
    4: "PenMarking",
    5: "Edge & Air Bubble",
    6: "Out of Focus",
    7: "Background",
}


def artifact(
    wsi: WSIData,
    tile_key: str,
    variants: Literal["grandqc_5x", "grandqc_7x", "grandqc_10x"] = "grandqc_7x",
    tissue_key: str = Key.tissue,
    batch_size: int = 16,
    n_workers: int = 0,
    device: str | None = None,
    key_added: str = "artifacts",
):
    if tissue_key not in wsi:
        raise ValueError(
            "Tissue segmentation is required before artifact segmentation."
            "Please run `pp.find_tissues` first."
        )

    if device is None:
        device = get_torch_device()

    model_mpp = {
        "grandqc_5x": 2,
        "grandqc_7x": 1.5,
        "grandqc_10x": 1,
    }

    mpp = model_mpp[variants]

    if tile_key is not None:
        # Check if the tile spec is compatible with the model
        spec = wsi.tile_spec(tile_key)
        if spec is None:
            raise ValueError(f"Tiles or tile spec for {tile_key} not found.")
        if spec.mpp != mpp:
            raise ValueError(
                f"Tile spec mpp {spec.mpp} is not "
                f"compatible with the model mpp {mpp}"
            )
        if spec.width != 512 or spec.height != 512:
            raise ValueError("Tile should be 512x512.")

    model = GrandQCArtifactSegmentation(model=variants.lstrip("grandqc_"))

    runner = SegmentationRunner(
        wsi,
        model,
        tile_key,
        transform=None,
        batch_size=batch_size,
        n_workers=n_workers,
        device=device,
        class_col="class",
        postprocess_kws={
            "ignore_index": [0, 1, 7],  # Ignore background, normal tissue
            "mapping": CLASS_MAPPING,
        },
    )
    arts = runner.run()
    add_shapes(wsi, key=key_added, shapes=arts)
