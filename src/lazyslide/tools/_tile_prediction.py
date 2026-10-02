from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from wsidata import WSIData
from wsidata.io import update_shapes_data

from lazyslide import _api
from lazyslide._const import Key
from lazyslide._utils import default_pbar

if TYPE_CHECKING:
    import torch
    from lazyslide_models import TilePredictionModelProtocol

    TP_MODEL = str | TilePredictionModelProtocol


def tile_prediction(
    wsi: WSIData,
    model: TP_MODEL,
    transform=None,
    batch_size: int = 16,
    num_workers: int = 0,
    prefetch_factor: int | None = None,
    tile_key: str = Key.tiles,
    amp: bool | None = None,
    autocast_dtype: torch.dtype = None,
    compile: bool | None = None,
    compile_kws: dict | None = None,
    device: str | None = None,
    pbar: bool = True,
):
    """
    Predict :term:`tiles <tile>` using a :term:`tile prediction model`.

    A list of available models can be listed with:

    .. code-block:: python

        from lazyslide_models import list_models

        list_models(task="tile_prediction")


    Parameters
    ----------
    wsi : :class:`WSIData <wsidata.WSIData>`
        The WSIData object to work on.
    model : str or TilePredictionModel
        The tile prediction model to use. If a string, it should be the name of the model.
    transform : callable, optional
        A :term:`transform function` to apply to the tiles before prediction. If None, the model's default transform is used.
    batch_size : int, default: 16
        The batch size for the DataLoader.
    num_workers : int, default: 0
        Number of worker threads for the DataLoader.
    prefetch_factor : int, optional
        The number of batches loaded in advance by each worker.
        Only used when :code:`num_workers > 0`.
    tile_key : str, default: "tiles"
        The key in the WSIData object where the tiles are stored.
    amp : bool, optional
        Whether to use automatic mixed precision.
    autocast_dtype : torch.dtype, optional
        The dtype for automatic mixed precision.
    compile : bool, optional
        Whether to compile the model with :func:`torch.compile`.
        Compilation is best-effort and is silently skipped for models
        that do not support it.
    compile_kws : dict, optional
        Keyword arguments passed to :func:`torch.compile`.
    device : str, optional
        The device to run the model on.
    pbar : bool, default: True
        Whether to show a progress bar during prediction.

    Returns
    -------
    None
        The predictions are added to the WSIData object.

    """
    import torch
    from torch.utils.data import DataLoader

    device = _api.default_value("device", device)

    is_cv_features = False
    if isinstance(model, str):
        from lazyslide_models import MODEL_REGISTRY
        from lazyslide_models.tile_prediction import CV_FEATURES

        if model == "spider":
            raise ValueError(
                "For spider model, please specify the variants, e.g. 'spider-breast'."
            )

        if model in CV_FEATURES:
            model = CV_FEATURES[model]()
            is_cv_features = True
        else:
            model = MODEL_REGISTRY[model]()
    model.to(device=device)
    model = _api.maybe_compile(model, compile, compile_kws)

    if transform is None:
        transform = model.get_transform()
    ds = wsi.ds.tile_images(tile_key=tile_key, transform=transform)

    loader_kws = _api.loader_kws(device, num_workers, prefetch_factor)
    non_blocking = loader_kws["pin_memory"]
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, **loader_kws)

    results = []

    with default_pbar(disable=not pbar) as progress_bar:
        task = progress_bar.add_task(
            f"Predicting tiles with {model.__class__.__name__}", total=len(ds)
        )

        amp_ctx = _api.autocast(device, amp, autocast_dtype)
        with amp_ctx, torch.inference_mode():
            for batch in dl:
                images = batch["image"]
                if not is_cv_features:
                    images = images.to(device, non_blocking=non_blocking)
                output = model.predict(images)
                results.append(pd.DataFrame(output))
                progress_bar.update(task, advance=len(images))
            progress_bar.refresh()
    # Concatenate all results
    results = pd.concat(results).reset_index(drop=True)

    # Add the predictions to the WSIData object
    update_shapes_data(wsi, tile_key, results)
