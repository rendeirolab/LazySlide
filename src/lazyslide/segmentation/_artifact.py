from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

from wsidata import WSIData
from wsidata.io import add_shapes

from lazyslide._utils import find_stack_level

from ._seg_runner import SemanticSegmentationRunner

if TYPE_CHECKING:
    import torch
    from lazyslide_models import SegmentationModelProtocol

# The GrandQC class names LazySlide has always written
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
    model: str | SegmentationModelProtocol = "grandqc",
    variant: str = "7x",
    mode: Literal["constant", "gaussian"] = "gaussian",
    sigma_scale: float = 0.125,
    low_memory: bool | None = None,
    threshold: float = 0.8,
    buffer_px: int = 2,
    batch_size: int = 4,
    num_workers: int = 0,
    device: str | None = None,
    amp: bool | None = None,
    autocast_dtype: torch.dtype = None,
    key_added: str = "artifacts",
    pbar: bool | None = None,
    *,
    prefetch_factor: int | None = None,
    compile: bool | None = None,
    compile_kws: dict | None = None,
):
    """
    :term:`Artifact segmentation` for the :term:`whole slide image <WSI>`.

    Run an artifact segmentation model on the whole slide image, by default
    GrandQC :cite:p:`Weng2024-jf`. GrandQC is trained on 512x512 tiles at
    mpp=2, 1.5 or 1 (variants 5x, 7x and 10x).

    GrandQC detects the following :term:`artifacts <artifact>`:

    - Fold
    - Darkspot & Foreign Object
    - Pen Marking
    - Edge & Air Bubble
    - Out of Focus

    Parameters
    ----------
    wsi : :class:`WSIData <wsidata.WSIData>`
        The :term:`WSIData` object to work on.
    tile_key : str
        The key of the tile table.
    model : str or SegmentationModelProtocol, default: "grandqc"
        The model to use for artifact segmentation: a model registry key (see
        :ref:`models-section`) or a model instance. "grandqc" is short for the
        "grandqc-artifact" key. A model that doesn't name its ``classes`` gets
        GrandQC's class names.
    variant : {"5x", "7x", "10x"}, default: "7x"
        The GrandQC variant. Only used for GrandQC.
    mode : {"constant", "gaussian"}, default: "gaussian"
        The probability distribution to apply for the prediction map.
        If "constant", uses uniform weights, "gaussian" applies a Gaussian weighting.
    sigma_scale : float, default: 0.125
        The scale of the Gaussian sigma for the importance map if mode is "gaussian".
    low_memory : bool, optional
        .. deprecated:: 0.13.0
            Has no effect and will be removed in 0.14.0.
    threshold : float, default: 0.8
        The probability threshold to consider a pixel as an artifact.
    buffer_px : int, default: 2
        The buffer in pixels to apply when merging :term:`polygons <polygon>`.
    batch_size : int, default: 4
        The batch size for :term:`segmentation`.
    num_workers : int, default: 0
        The number of workers for data loading.
    device : str, default: None
        The device for the model.
    amp : bool, optional
        Whether to use automatic mixed precision.
    autocast_dtype : torch.dtype, optional
        The dtype for automatic mixed precision.
    key_added : str, default: "artifacts"
        The key for the added artifact shapes.
    pbar : bool, optional
        Whether to show a progress bar during segmentation.
    prefetch_factor : int, optional
        The number of batches loaded in advance by each worker.
        Only used when :code:`num_workers > 0`.
    compile : bool, optional
        Whether to compile the model with :func:`torch.compile`.
        Compilation is best-effort and is silently skipped for models
        that do not support it.
    compile_kws : dict, optional
        Keyword arguments passed to :func:`torch.compile`.

    Returns
    -------
    None
        The artifact shapes are added to the :bdg-danger:`shapes` slot
        of the WSIData object.

    """
    from lazyslide_models import MODEL_REGISTRY
    from lazyslide_models.segmentation import GrandQCArtifact

    if model in ("grandqc", "grandqc-artifact"):
        model = MODEL_REGISTRY["grandqc-artifact"](variant=variant)
    elif isinstance(model, str):
        model = MODEL_REGISTRY[model]()
    is_grandqc = isinstance(model, GrandQCArtifact)

    spec = wsi.tile_spec(tile_key)
    if spec is None:
        raise ValueError(f"Tiles or tile spec for {tile_key} not found.")
    if is_grandqc:
        # ponytail: GrandQC's input mpp per variant is kept here until
        # rendeirolab/lazyslide-models#37 lets the model declare it.
        mpp = {"5x": 2, "7x": 1.5, "10x": 1}[variant]
        if spec.mpp != mpp:
            raise ValueError(
                f"Tile spec mpp {spec.mpp} is not compatible with the model mpp {mpp}"
            )
        if spec.width != 512 or spec.height != 512:
            raise ValueError("Tile should be 512x512.")
    if spec.overlap_x == 0 or spec.overlap_y == 0:
        mode = "constant"
        warnings.warn(
            "The tiles has no overlap, using constant mode instead. "
            "Please consider rerun pp.tile_tissue to create overlapping tiles.",
            stacklevel=find_stack_level(),
        )
    # GrandQC keeps the class names LazySlide has always written
    classes = None if is_grandqc else getattr(model, "classes", None)

    runner = SemanticSegmentationRunner(
        wsi=wsi,
        model=model,
        tile_key=tile_key,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        device=device,
        amp=amp,
        autocast_dtype=autocast_dtype,
        compile=compile,
        compile_kws=compile_kws,
        mode=mode,
        sigma_scale=sigma_scale,
        low_memory=low_memory,
        threshold=threshold,
        buffer_px=buffer_px,
        class_names=classes or CLASS_MAPPING,
        pbar=pbar,
    )
    arts = runner.run()
    arts = arts[~arts["class"].isin(["Background", "Normal Tissue"])]
    arts = arts.explode().reset_index(drop=True)
    if len(arts) == 0:
        print("No artifacts detected.")
        return

    add_shapes(wsi, key=key_added, shapes=arts)
