from __future__ import annotations

from typing import Literal

from wsidata import WSIData
from wsidata.io import add_shapes

from lazyslide._const import Key
from lazyslide._utils import get_torch_device
from ._seg_runner import SegmentationRunner
from ..models.segmentation import GrandQCArtifact

# Define class mapping
CLASS_MAPPING = {
    1: "Normal Tissue",
    2: "Fold",
    3: "Dark spot & Foreign Object",
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
    batch_size: int = 4,
    num_workers: int = 0,
    device: str | None = None,
    key_added: str = "artifacts",
):
    """
    Artifact segmentation for the whole slide image.

    Run GrandQC artifact segmentation model on the whole slide image.
    The model is trained on 512x512 tiles with mpp=1.5, 2, or 1.

    It can detect the following artifacts:
    - Fold
    - Darkspot & Foreign Object
    - Pen Marking
    - Edge & Air Bubble
    - Out of Focus

    Parameters
    ----------
    wsi : WSIData
        The whole slide image data.
    tile_key : str
        The key of the tile table.
    variants : {"grandqc_5x", "grandqc_7x", "grandqc_10x"}, default: "grandqc_7x"
        The model variant to use for segmentation.
    tissue_key : str, default: Key.tissue
        The key of the tissue table.
    batch_size : int, default: 4
        The batch size for segmentation.
    num_workers : int, default: 0
        The number of workers for data loading.
    device : str, default: None
        The device for the model.
    key_added : str, default: "artifacts"
        The key for the added artifact shapes.

    """
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

    model = GrandQCArtifact(model=variants.lstrip("grandqc_"))

    runner = SegmentationRunner(
        wsi,
        model,
        tile_key,
        transform=None,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        class_col="class",
        postprocess_kws={
            "ignore_index": [0, 1, 7],  # Ignore background, normal tissue
            "mapping": CLASS_MAPPING,
        },
    )
    arts = runner.run()
    add_shapes(wsi, key=key_added, shapes=arts)
