from __future__ import annotations

import warnings
from typing import Literal

from wsidata import WSIData
from wsidata.io import add_shapes

from lazyslide._utils import find_stack_level, get_torch_device

from ..models.segmentation import GrandQCArtifact
from ._seg_runner import SemanticSegmentationRunner

# Define class mapping
CLASS_MAPPING = {
    0: "Background",
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
    model: str = "grandqc",
    variant: str = "7x",
    mode: Literal["constant", "gaussian"] = "gaussian",
    sigma_scale: float = None,
    low_memory: bool = False,
    threshold: float = 0.8,
    buffer_px: int = 2,
    batch_size: int = 4,
    num_workers: int = 0,
    device: str | None = None,
    key_added: str = "artifacts",
    *args,
):
    """
    Artifact segmentation for the whole slide image.

    Run GrandQC :cite:p:`Weng2024-jf` artifact segmentation model on the whole slide image.
    The model is trained on 512x512 tiles with mpp=1.5, 2, or 1.

    It can detect the following artifacts:

    - Fold
    - Darkspot & Foreign Object
    - Pen Marking
    - Edge & Air Bubble
    - Out of Focus

    Parameters
    ----------
    wsi : :class:`WSIData <wsidata.WSIData>`
        The WSIData object to work on.
    tile_key : str
        The key of the tile table.
    model : {"grandqc"}, default: "grandqc"
        The model to use for artifact segmentation.
    variant : str, default: "7x"
        The model variants, grandqc has variants 5x, 7x and 10x.
    mode : {"constant", "gaussian"}, default: "gaussian"
        The probability distribution to apply for the prediction map.
        If "constant", uses uniform weights, "gaussian" applies a Gaussian weighting.
    sigma_scale : float
        The scale of the Gaussian sigma for the importance map if mode is "gaussian".
        If None, the scale is calculated based on the overlap of the tiles.
    low_memory : bool, default: False
        Whether to use a low-memory mode for processing large slides.
    threshold : float, default: 0.8
        The probability threshold to consider a pixel as an artifact.
    buffer_px : int, default: 2
        The buffer in pixels to apply when merging polygons.
    batch_size : int, default: 4
        The batch size for segmentation.
    num_workers : int, default: 0
        The number of workers for data loading.
    device : str, default: None
        The device for the model.
    key_added : str, default: "artifacts"
        The key for the added artifact shapes.

    """

    if "variants" in args:
        warnings.warn(
            "`variants` is deprecated. Use `model` and `variant` instead.",
            stacklevel=find_stack_level(),
        )

    if device is None:
        device = get_torch_device()

    model_mpp = {
        "5x": 2,
        "7x": 1.5,
        "10x": 1,
    }

    mpp = model_mpp[variant]

    if tile_key is not None:
        # Check if the tile spec is compatible with the model
        spec = wsi.tile_spec(tile_key)
        if spec is None:
            raise ValueError(f"Tiles or tile spec for {tile_key} not found.")
        if spec.mpp != mpp:
            raise ValueError(
                f"Tile spec mpp {spec.mpp} is not compatible with the model mpp {mpp}"
            )
        if spec.width != 512 or spec.height != 512:
            raise ValueError("Tile should be 512x512.")
        if sigma_scale is None:
            sigma_scale = spec.overlap_y * 0.5  # simple heuristic
        if spec.overlap_x == 0 or spec.overlap_y == 0:
            mode = "constant"
            warnings.warn(
                "The tiles has no overlap, using constant mode instead. "
                "Please consider rerun pp.tile_tissue to create overlapping tiles."
            )

    model = GrandQCArtifact(variant=variant)

    runner = SemanticSegmentationRunner(
        wsi=wsi,
        model=model,
        tile_key=tile_key,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        mode=mode,
        sigma_scale=sigma_scale,
        low_memory=low_memory,
        threshold=threshold,
        buffer_px=buffer_px,
        class_names=CLASS_MAPPING,
    )
    arts = runner.run()
    arts = arts[~arts["class"].isin(["Background", "Normal Tissue"])]
    arts = arts.explode().reset_index(drop=True)
    if len(arts) == 0:
        print("No artifacts detected.")
        return

    add_shapes(wsi, key=key_added, shapes=arts)
