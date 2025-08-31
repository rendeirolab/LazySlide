from __future__ import annotations

import warnings

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

    model = GrandQCArtifact(variant=variant)

    runner = SemanticSegmentationRunner(
        wsi=wsi,
        model=model,
        tile_key=tile_key,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        class_names=CLASS_MAPPING,
    )
    arts = runner.run()
    arts = arts.explode().reset_index(drop=True)
    add_shapes(wsi, key=key_added, shapes=arts)
