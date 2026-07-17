from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from wsidata import WSIData
from wsidata.io import add_features

from lazyslide import _api
from lazyslide._const import Key
from lazyslide._utils import default_pbar

if TYPE_CHECKING:
    import torch
    from anndata import AnnData
    from lazyslide_models.base import FeaturePredictionModelProtocol


def feature_prediction(
    wsi: WSIData,
    model: str | FeaturePredictionModelProtocol,
    feature_key: str | None = None,
    *,
    batch_size: int = 1024,
    tile_key: str = Key.tiles,
    key_added: str | None = None,
    amp: bool | None = None,
    autocast_dtype: torch.dtype | None = None,
    device: str | None = None,
    pbar: bool | None = None,
) -> AnnData:
    """Predict tile-level values from an existing feature matrix.

    Parameters
    ----------
    wsi : :class:`WSIData <wsidata.WSIData>`
        The whole-slide image object containing tile features.
    model : str or feature prediction model
        A registered feature prediction model name or an object implementing
        ``predict(features)``.
    feature_key : str, optional
        Feature table used as model input. When omitted, this is inferred from
        ``model.features_model_name``.
    batch_size : int, default: 1024
        Number of tile feature vectors passed to the model at once.
    tile_key : str, default: "tiles"
        Tile table associated with the input and output features.
    key_added : str, optional
        Key used to store the prediction table. Defaults to
        ``{model_name}_{tile_key}``.
    amp : bool, optional
        Whether to use automatic mixed precision.
    autocast_dtype : torch.dtype, optional
        Data type used for automatic mixed precision.
    device : str, optional
        Device on which to run inference.
    pbar : bool, optional
        Whether to display a progress bar.

    Returns
    -------
    None

    Notes
    -----
    Dense input matrices are passed to the model as basic row slices. NumPy
    therefore supplies views rather than copying the full feature vectors.
    """
    import torch
    from lazyslide_models.base import FeaturePredictionModelProtocol

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    amp = _api.default_value("amp", amp)
    autocast_dtype = _api.default_value("autocast_dtype", autocast_dtype)
    device = _api.default_value("device", device)
    pbar = _api.default_value("pbar", pbar)

    model_name = model if isinstance(model, str) else getattr(model, "name", None)
    if isinstance(model, str):
        from lazyslide_models import MODEL_REGISTRY

        model_cls = MODEL_REGISTRY.get(model)
        if model_cls is None:
            raise ValueError(f"Feature prediction model '{model}' was not found.")
        model = model_cls()

    if not isinstance(model, FeaturePredictionModelProtocol):
        raise TypeError(
            "model must be a model name or implement FeaturePredictionModelProtocol."
        )

    if model_name is None:
        model_name = model.__class__.__name__

    if feature_key is None:
        feature_key = getattr(model, "features_model_name", None)
        if feature_key is None:
            raise ValueError(
                "feature_key is required when the model does not define "
                "features_model_name."
            )
    feature_key = wsi._check_feature_key(feature_key, tile_key)
    features = wsi.tables[feature_key].X

    if features.shape[0] != len(wsi.shapes[tile_key]):
        raise ValueError(
            f"Feature table '{feature_key}' has {features.shape[0]} rows, but "
            f"tile table '{tile_key}' has {len(wsi.shapes[tile_key])} rows."
        )

    try:
        model.to(device)
    except (AttributeError, TypeError):
        pass

    def to_numpy(value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        value = np.asarray(value)
        if value.ndim == 2 and value.shape[1] == 1:
            value = value[:, 0]
        if value.ndim != 1:
            raise ValueError(
                "Each feature prediction output must have shape (n_tiles,) "
                "or (n_tiles, 1)."
            )
        return value

    results: dict[str, list[np.ndarray]] = {}
    output_names: tuple[str, ...] | None = None
    n_obs = features.shape[0]
    amp_device = device.type if isinstance(device, torch.device) else device
    amp_ctx = torch.autocast(amp_device, dtype=autocast_dtype) if amp else nullcontext()

    with default_pbar(disable=not pbar) as progress_bar:
        task = progress_bar.add_task(
            f"Predicting features with {model_name}", total=n_obs
        )
        with amp_ctx, torch.inference_mode():
            for start in range(0, n_obs, batch_size):
                stop = min(start + batch_size, n_obs)
                output = model.predict(features[start:stop])
                if not isinstance(output, Mapping) or not output:
                    raise TypeError("model.predict must return a non-empty mapping.")

                names = tuple(output)
                if output_names is None:
                    output_names = names
                    results = {name: [] for name in names}
                elif names != output_names:
                    raise ValueError(
                        "model.predict returned inconsistent output names between batches."
                    )

                for name, value in output.items():
                    array = to_numpy(value)
                    if len(array) != stop - start:
                        raise ValueError(
                            f"Output '{name}' has {len(array)} rows for a batch "
                            f"of {stop - start}."
                        )
                    results[name].append(array)
                progress_bar.update(task, advance=stop - start)
        progress_bar.refresh()

    if output_names is None:
        raise ValueError("Cannot run feature prediction on an empty feature table.")

    predictions = np.column_stack(
        [np.concatenate(results[name]) for name in output_names]
    )
    key_added = key_added or Key.feature(str(model_name), tile_key)
    add_features(
        wsi,
        key=key_added,
        tile_key=tile_key,
        features=predictions,
        var=pd.DataFrame(index=pd.Index(output_names, dtype=str)),
    )
