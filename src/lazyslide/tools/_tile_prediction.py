from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING, Union

import pandas as pd
import torch
from torch.utils.data import DataLoader
from wsidata import WSIData
from wsidata.io import update_shapes_data

from lazyslide._const import Key
from lazyslide._utils import default_pbar, get_torch_device
from lazyslide.models.base import TilePredictionModel

if TYPE_CHECKING:
    TP_MODEL = Union[str, TilePredictionModel]


def tile_prediction(
    wsi: WSIData,
    model: TP_MODEL,
    transform=None,
    batch_size: int = 16,
    num_workers: int = 0,
    tile_key: str = Key.tiles,
    amp: bool = False,
    autocast_dtype: torch.dtype = torch.float16,
    device: str | None = None,
    pbar: bool = True,
):
    """
    Predict tiles using a tile prediction model.

    Parameters
    ----------
    wsi : :class:`WSIData <wsidata.WSIData>`
        The WSIData object to work on.
    model : str or TilePredictionModel
        The tile prediction model to use. If a string, it should be the name of the model.
    transform : callable, optional
        A transform to apply to the tiles before prediction. If None, the model's default transform is used.
    batch_size : int, default: 16
        The batch size for the DataLoader.
    num_workers : int, default: 0
        Number of worker threads for the DataLoader.
    tile_key : str, default: "tiles"
        The key in the WSIData object where the tiles are stored.
    device : str, optional
        The device to run the model on.
    pbar : bool, default: True
        Whether to show a progress bar during prediction.

    Returns
    -------
    None
        The predictions are added to the WSIData object.

    """
    is_cv_features = False
    if isinstance(model, str):
        from lazyslide.models import MODEL_REGISTRY
        from lazyslide.models.tile_prediction import CV_FEATURES

        if model == "spider":
            raise ValueError(
                "For spider model, please specify the variants, e.g. 'spider-breast'."
            )

        if model in CV_FEATURES:
            model = CV_FEATURES[model]()
            is_cv_features = True
        else:
            card = MODEL_REGISTRY[model]
            if card is None:
                raise ValueError(f"Model '{model}' not found in the registry.")
            model = card()

    if device is None:
        device = get_torch_device()
    model.to(device=device)

    if transform is None:
        transform = model.get_transform()
    ds = wsi.ds.tile_images(tile_key=tile_key, transform=transform)

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    results = []

    with default_pbar(disable=not pbar) as progress_bar:
        if isinstance(model, str):
            model_name = model
        else:
            model_name = model.__class__.__name__
        task = progress_bar.add_task(
            f"Predicting tiles with {model_name}", total=len(ds)
        )

        amp_ctx = torch.autocast(device, autocast_dtype) if amp else nullcontext()
        with amp_ctx, torch.inference_mode():
            for batch in dl:
                images = batch["image"]
                if not is_cv_features:
                    images = images.to(device)
                output = model.predict(images)
                results.append(pd.DataFrame(output))
                progress_bar.update(task, advance=len(images))
            progress_bar.refresh()
    # Concatenate all results
    results = pd.concat(results).reset_index(drop=True)

    # Add the predictions to the WSIData object
    update_shapes_data(wsi, tile_key, results)


def _get_model(model: TP_MODEL) -> TilePredictionModel:
    """
    Get the tile prediction model from a string or a TilePredictionModel instance.

    Parameters
    ----------
    model : str or TilePredictionModel
        The model to get.

    Returns
    -------
    TilePredictionModel
        The tile prediction model instance.

    """
    if isinstance(model, str):
        from lazyslide.models import MODEL_REGISTRY
        from lazyslide.models.tile_prediction import CV_FEATURES

        if model in CV_FEATURES:
            return CV_FEATURES[model]()

        card = MODEL_REGISTRY.get(model)
        if card is None:
            raise ValueError(f"Model '{model}' not found in the registry.")
        return card()
    elif isinstance(model, TilePredictionModel):
        return model
    else:
        raise TypeError(
            f"Cannot recognize {model}, "
            f"model must be a string or a TilePredictionModel instance."
        )
