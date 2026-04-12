from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Literal, Sequence

import geopandas as gpd
import numpy as np
import torch
from shapely import box
from torch.utils.data import DataLoader
from wsidata import TileSpec, WSIData
from wsidata.io import add_features

import lazyslide._api as _api
from lazyslide._const import Key
from lazyslide._utils import default_pbar
from lazyslide.models import (
    MODEL_REGISTRY,
    ImageModel,
    ImageModelProtocol,
    ModelBaseProtocol,
    ViTModelProtocol,
    list_models,
)
from lazyslide.preprocess._tiles import _add_tiles


def load_models(model_name: str, dense=False, model_path=None, token=None, **kwargs):
    """Load a model with timm or torch.hub.load"""

    if model_name in MODEL_REGISTRY:
        model = MODEL_REGISTRY[model_name](model_path=model_path, token=token, **kwargs)
    else:
        from lazyslide.models import TimmModel, TimmViTModel

        if dense:
            try:
                model = TimmViTModel(model_name, token=token, **kwargs)
            except ValueError:
                raise ValueError(f"Model {model_name} is not a ViT model.")
        else:
            model = TimmModel(model_name, token=token, **kwargs)

    return model, model_name


def to_numpy(t):
    return t if isinstance(t, np.ndarray) else t.cpu().numpy()


DEFAULT_POOL_MODE = {
    "h0-mini": "cls_patch_mean",
    "h-optimus-0": "cls_patch_mean",
    "h-optimus-1": "cls_patch_mean",
    "midnight": "cls_patch_mean",
    "gigapath": "cls",
    "uni": "cls",
    "uni2": "cls",
}


# TODO: Test if it's possible to load model files
# TODO: Add color normalization
def feature_extraction(
    wsi: WSIData,
    model: str | Callable | ImageModel = None,
    *,
    model_path: str | Path = None,
    model_name: str = None,
    jit: bool = False,
    token: str = None,
    load_kws: dict = None,
    transform: Callable = None,
    # For inference
    device: str = None,
    amp: bool = None,
    autocast_dtype: torch.dtype = None,
    batch_size: int = 32,
    num_workers: int = 0,
    pbar: bool = None,
    # For input
    tile_key: str = Key.tiles,
    dense: bool = False,
    pool_mode: Literal["cls", "cls_patch_mean"] | None = None,
    # For results
    key_added: str = None,
    return_features: bool = False,
    **kwargs,
):
    """
    Extract :term:`features` from :term:`WSI` :term:`tiles <tile>` using a pre-trained :term:`vision models <vision model>`.

    To list all timm models:

    .. code-block:: python

        >>> import timm
        >>> timm.list_models(pretrained=True)

    To list all lazyslide built-in models:

    .. code-block:: python

        >>> import lazyslide as zs
        >>> zs.models.list_models()

    Parameters
    ----------
    wsi : :class:`WSIData <wsidata.WSIData>`
        The whole-slide image object.
    model : str or model object
        The model used for image :term:`feature extraction`.
        A list of built-in :term:`foundation models <foundation model>` can be found in :ref:`models-section`.
        Other models can be loaded from :term:`Hugging Face`, but only models with feature extraction head implemented.
    model_path : str or Path
        The path to the model file. Either model or model_path must be provided.
        If you don't have internet access, you can download the model file and load it from the local path.
        You can also load custom models from local files.
    model_name : str, optional
        If you provide your own model, you can specify the model name for the key_added.
        Or you can override the model name by providing a new model name.
    jit : bool, default: False
        Whether the model is a JIT model. If True, use torch.jit.load to load the model.
    token : str, optional
        The token for downloading the model from Hugging Face Hub for foundation models.
    load_kws : dict, optional
        Options to pass to the model creation function.
    transform : callable, optional
        The :term:`transform function` for the input image.
        If not provided, a default ImageNet transform function will be used.
    device : str, optional
        The device to use for inference. If not provided, the device will be automatically selected.
    amp : bool, default: False
        Whether to use automatic mixed precision.
    autocast_dtype : torch.dtype, default: torch.float16
        The dtype for automatic mixed precision.
    batch_size : int, optional
        The batch size for inference.
    num_workers : int, optional
        The number of workers for data loading.
    pbar : bool, default: True
        Whether to show progress bar.
    tile_key : str, default: 'tiles'
        The key of the tiles dataframe in the spatial data object.
    dense : bool, default: False
        Whether to extract dense features for ViT models.
    pool_mode : {'cls', 'cls_patch_mean'}, optional
        The pooling mode for dense features.
    key_added : str, optional
        The key to store the extracted features.
    return_features : bool, default: False
        Whether to return the extracted features.

    Returns
    -------
    :class:`numpy.ndarray` or None
        If return_features is True, return the extracted features.

    .. note::
        The feature matrix will be added to :code:`{model_name}_{tile_key}`
        in :bdg-danger:`tables` slot of :term:`WSIData` object.

    Examples
    --------

    .. code-block:: python

        >>> import lazyslide as zs
        >>> wsi = zs.datasets.sample()
        >>> zs.pp.find_tissues(wsi)
        >>> zs.pp.tile_tissues(wsi, 256, mpp=0.5)
        >>> zs.tl.feature_extraction(wsi, "resnet50")
        >>> wsi.fetch.features_anndata("resnet50")

    """

    device = _api.default_value("device", device)
    amp = _api.default_value("amp", amp)
    autocast_dtype = _api.default_value("autocast_dtype", autocast_dtype)
    pbar = _api.default_value("pbar", pbar)

    load_kws = {} if load_kws is None else load_kws

    if model is not None:
        if isinstance(model, Callable):
            model = model
        elif isinstance(model, str):
            model, default_model_name = load_models(
                dense=dense,
                model_name=model,
                model_path=model_path,
                token=token,
                **load_kws,
            )
            if model_name is None:
                model_name = default_model_name
        elif isinstance(model, ImageModelProtocol):
            model = model
            model_name = model.name
        else:
            raise ValueError("Model must be a model name or a model object.")
    else:
        if model_path is None:
            raise ValueError("Either model or model_path must be provided.")
        model_path = Path(model_path)
        if model_path.exists():
            load_kws.setdefault("weights_only", False)
            load_func = torch.load if not jit else torch.jit.load
            model = load_func(model_path, **load_kws)
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
    # Deal with key_added
    if key_added is None:
        if model_name is not None:
            key_added = model_name
        elif isinstance(model, ImageModelProtocol):
            key_added = model.name
        elif hasattr(model, "__class__"):
            key_added = model.__class__.__name__
        elif hasattr(model, "__name__"):
            key_added = model.__name__
        else:
            key_added = "features"
        key_added = Key.feature(key_added, tile_key)
    try:
        model.to(device)
    except:  # noqa: E722
        pass

    if transform is None:
        if isinstance(model, ModelBaseProtocol):
            transform = model.get_transform()

    n_tiles = len(wsi.shapes[tile_key])

    # Setup dense features tiles
    if dense:
        if isinstance(model, ViTModelProtocol):
            token_tiles, token_tiles_spec = subdivide_tiles(
                wsi, model.grid_size, tile_key
            )
        else:
            raise NotImplementedError(
                "Dense features are only supported for ViT models."
            )
        if pool_mode is None:
            pool_mode = DEFAULT_POOL_MODE.get(model_name, "cls")
        if pool_mode not in ["cls", "cls_patch_mean"]:
            raise ValueError(f"Invalid pool_mode: {pool_mode}")

    with default_pbar(disable=not pbar) as progress_bar:
        task = progress_bar.add_task("Extracting features", total=n_tiles)
        dataset = wsi.ds.tile_images(tile_key=tile_key, transform=transform)
        loader = DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, **kwargs
        )
        # Extract features
        dense_features = []
        features = []
        if isinstance(device, torch.device):
            device = device.type
        amp_ctx = torch.autocast(device, autocast_dtype) if amp else nullcontext()
        with amp_ctx, torch.inference_mode():
            for batch in loader:
                image = batch["image"].to(device)
                if dense and isinstance(model, ViTModelProtocol):
                    intermediate_features = model.encode_image_dense(image)
                    cls_feature = intermediate_features[:, 0]
                    patch_token_feature = intermediate_features[
                        :, model.num_prefix_tokens :
                    ]
                    patch_mean = patch_token_feature.mean(1)
                    # Keep (batch_size, n_patches, embed_dim) — flatten per-tile later to preserve order
                    dense_feature = patch_token_feature.reshape(
                        -1, patch_token_feature.shape[-1]
                    )

                    if pool_mode == "cls":
                        output = cls_feature
                    elif pool_mode == "cls_patch_mean":
                        output = torch.cat([cls_feature, patch_mean], dim=-1)
                    else:
                        raise ValueError(f"Invalid pool_mode: {pool_mode}")
                    dense_features.append(to_numpy(dense_feature))
                elif isinstance(model, ImageModelProtocol):
                    output = model.encode_image(image)
                elif callable(model):
                    output = model(image)
                else:
                    raise TypeError(
                        f"Model {type(model)} is not callable and does not have encode_image method."
                    )
                features.append(to_numpy(output))
                progress_bar.update(task, advance=len(image))
                del batch  # Free up memory
        # The progress bar may not reach 100% if exit too early
        # Force update
        progress_bar.refresh()
        features = np.vstack(features)

    add_features(wsi, key=key_added, tile_key=tile_key, features=features)
    if dense:
        _add_tiles(wsi, token_tiles, token_tiles_spec, key_added=f"{tile_key}_dense")
        dense_features = np.vstack(dense_features)
        add_features(
            wsi,
            key=f"{key_added}_dense",
            tile_key=f"{tile_key}_dense",
            features=dense_features,
        )
    if return_features:
        return features
    return None


def feature_aggregation(
    wsi: WSIData,
    feature_key: str,
    layer_key: str = None,
    encoder: str | Callable = "mean",
    tile_key: str = Key.tiles,
    by: str | Sequence[str] | None = None,
    agg_key: str = None,
    amp: bool = False,
    autocast_dtype: torch.dtype = torch.float16,
    device: str = "cpu",
):
    """
    Aggregate :term:`features` by groups.

    The :term:`feature aggregation` is done by applying an encoder to a group of features to acquire
    a 1d representation of the group. Notice that the final shape of the aggregated
    features might not be the same as the original features.

    Parameters
    ----------
    wsi : :class:`WSIData <wsidata.WSIData>`
        The whole-slide image object.
    feature_key : str
        The key to indicate which feature to aggregate.
    layer_key : str, optional
        The key of the layer in the feature table.
    encoder : str or callable, default: 'mean'

        - Numpy functions: 'mean', 'median', 'sum', 'std', 'var', ...
        - :code:`prism`: Prism slide encoder. The feature must be extracted by :code:`Virchow` model.
        - :code:`titan`: Titan slide encoder. The feature must be extracted by :code:`Titan`/:code:`CONCH_v1.5` model.
        - :code:`madeleine`: Madeleine slide encoder. The feature must be extracted by :code:`CONCH` model.
        - :code:`chief`: Chief slide encoder. The feature must be extracted by :code:`CHIEF` model.
    tile_key : str, default: 'tiles'
        The key of the tiles dataframe in the spatial data object.
    by : str or array of str, default: None
        The level to aggregate the features.

        - By default will aggregate the features from all tiles in the slide.
        - Column name in tile dataframe: Aggregate the features by specific column.
          For example, to aggregate by tissue pieces, set by='tissue_id'.
    agg_key : str, optional
        The key to store the aggregated features. If not provided, the key will be 'agg_{by}'.
    amp : bool, default: False
        Whether to use automatic mixed precision.
    autocast_dtype : torch.dtype, default: torch.float16
        The dtype for automatic mixed precision.
    device : str, optional
        The device to use for inference. If not provided, the device will be automatically selected.

    Returns
    -------
    None

    .. note::

        The aggregation features and operation will be recorded in the :bdg-warning:`uns` slot of features AnnData.
        The aggregated features will only be added to the :bdg-warning:`varm` slot if
        their shape is the same as the original features.

    Examples
    --------
    .. code-block:: python

        >>> import lazyslide as zs
        >>> wsi = zs.datasets.sample(with_data=False)
        >>> zs.pp.find_tissues(wsi)
        >>> zs.pp.tile_tissues(wsi, 256, mpp=0.5)
        >>> zs.tl.feature_extraction(wsi, "resnet50")
        >>> zs.tl.feature_aggregation(wsi, feature_key="resnet50", by="tissue_id")
        >>> wsi.tables['resnet50_tiles'].uns['agg_tissue_id']

    """
    device = _api.default_value("device", device)
    amp = _api.default_value("amp", amp)
    autocast_dtype = _api.default_value("autocast_dtype", autocast_dtype)

    tiles_table = wsi.shapes[tile_key]
    tile_spec = wsi.tile_spec(tile_key)
    coords = tiles_table.bounds[["minx", "miny"]]
    feature_key = wsi._check_feature_key(feature_key, tile_key)
    if layer_key is None:
        features = wsi.tables[feature_key].X
    else:
        features = wsi.tables[feature_key].layers[layer_key]

    agg_info = {"encoder": encoder, "tile_key": tile_key}

    if by is None:
        if agg_key is None:
            agg_key = "agg_slide"
        slide_reprs = _encode_slide(
            features,
            encoder,
            coords,
            device=device,
            amp=amp,
            autocast_dtype=autocast_dtype,
            tile_spec=tile_spec,
        )
        agg_fs = slide_reprs["features"]
        for k, v in slide_reprs.items():
            if k != "features":
                agg_info[k] = v
    else:
        if isinstance(by, str):
            by = [by]
        if agg_key is None:
            agg_key = f"agg_{'_'.join(by)}"
        agg_fs = []
        agg_latents = []
        agg_annos = []
        for annos, x in tiles_table.groupby(by):
            slide_reprs = _encode_slide(
                features[x.index],
                encoder,
                coords.loc[x.index],
                device=device,
                amp=amp,
                autocast_dtype=autocast_dtype,
                tile_spec=tile_spec,
            )
            agg_fs.append(slide_reprs["features"])
            if "latents" in slide_reprs:
                agg_latents.append(slide_reprs["latents"])
            agg_annos.append(list(annos))
        agg_fs = np.vstack(agg_fs)

        agg_info["keys"] = by
        agg_info["values"] = agg_annos
        if len(agg_latents) > 0:
            agg_latents = np.vstack(agg_latents)
            agg_info["latents"] = agg_latents
        # return agg_fs, agg_annos

    # The aggregated features should have the same number of columns as the original features
    # Otherwise, we have to write it to uns
    feature_table = wsi.tables[feature_key]
    agg_info["features"] = agg_fs
    if agg_fs.shape[1] == features.shape[1]:
        # Add the aggregated features to varm slot of the feature table
        feature_table.varm[agg_key] = agg_fs.T

    # Add the aggregation operation to the uns slot
    agg_ops = feature_table.uns.get("agg_ops", {})
    agg_ops[agg_key] = agg_info
    feature_table.uns["agg_ops"] = agg_ops


def _encode_slide(
    features,
    encoder,
    coords=None,
    amp: bool = False,
    autocast_dtype: torch.dtype = torch.float16,
    device=None,
    tile_spec=None,
):
    """
    Encode slide features using various methods.

    Parameters
    ----------
    features : numpy.ndarray
        Feature matrix with shape (n_tiles, n_features)
    encoder : str or callable
        Encoding method to use
    coords : pandas.DataFrame, optional
        Tile coordinates with columns 'minx' and 'miny'
    device : str, optional
        Device to use for PyTorch operations
    tile_spec : object, optional
        Tile specification object with base_width attribute

    Returns
    -------
    dict
        Dictionary with 'features' key containing the encoded features
        and optionally other keys like 'latents'
    """
    result_dict = {"features": None}

    # Simple statistical aggregation methods
    # Model-based encoding methods
    slide_encoders = set(list_models("slide_encoder"))
    slide_encoders.update(("chief", "gigapath"))
    if encoder in slide_encoders:
        # Convert features and coordinates to PyTorch tensors
        fs = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
        cs = torch.tensor(coords.values, dtype=torch.long).unsqueeze(0).to(device)
        amp_ctx = nullcontext() if not amp else torch.autocast(device, autocast_dtype)
        with amp_ctx, torch.inference_mode():
            if encoder in {"chief", "chief-slide-encoder"}:
                key = "chief-slide-encoder"
            elif encoder in {"gigapath", "gigapath-slide-encoder"}:
                key = "gigapath-slide-encoder"
            else:
                key = encoder
            model = MODEL_REGISTRY[key]()
            model.to(device)
            if encoder == "prism":
                slide_reprs = model.encode_slide(fs)
                agg_features = slide_reprs["image_embedding"].cpu().detach().numpy()
                img_latents = slide_reprs["image_latents"].cpu().detach().numpy()
                result_dict["latents"] = img_latents
            elif encoder == "titan":
                agg_features = model.encode_slide(fs, cs, tile_spec.base_width)
            else:
                agg_features = model.encode_slide(fs, cs)

    # Unknown encoder
    else:
        func = getattr(np, encoder, None)
        if callable(func):
            # Use numpy function for aggregation
            agg_features = func(features, axis=0)
        else:
            raise ValueError(f"Unknown slide encoding method: {encoder}")

    # Ensure the features have the right shape (batch, features)
    if agg_features.ndim == 1:
        agg_features = agg_features.reshape(1, -1)
    if isinstance(agg_features, torch.Tensor):
        agg_features = agg_features.detach().cpu().numpy()

    result_dict["features"] = agg_features
    return result_dict


def subdivide_tiles(
    wsi: WSIData,
    subdivisions: int | tuple[int, int],
    tile_key: str = Key.tiles,
) -> tuple[gpd.GeoDataFrame, TileSpec]:
    """
    Subdivide tiles into smaller tiles.

    Parameters
    ----------
    wsi : :class:`WSIData <wsidata.WSIData>`
        The whole-slide image object.
    subdivisions : int or tuple of int
        The number of subdivisions in each dimension.
        If int, the same number of subdivisions will be used for both width and height.
        If tuple, (nw, nh) subdivisions will be used.
    tile_key : str, default: 'tiles'
        The key of the tiles dataframe in the spatial data object.

    Returns
    -------
    tuple[geopandas.GeoDataFrame, TileSpec]
        A tuple containing the subdivided tiles as a GeoDataFrame and the
        corresponding TileSpec for the generated sub-tiles.

    """
    if isinstance(subdivisions, int):
        nw = nh = subdivisions
    else:
        nw, nh = subdivisions

    tiles_table = wsi.shapes[tile_key]
    tile_spec = wsi.tile_spec(tile_key)

    # Use the actual geometry bounds to get base-level coordinate dimensions.
    # tile_spec.width/height are in pixel space at the requested mpp, which may
    # differ from the slide base coordinate space when the tile is downsampled.
    sample_bounds = tiles_table.iloc[0].geometry.bounds  # (minx, miny, maxx, maxy)
    base_tile_w = sample_bounds[2] - sample_bounds[0]
    base_tile_h = sample_bounds[3] - sample_bounds[1]
    new_width = base_tile_w / nw
    new_height = base_tile_h / nh

    new_tiles = []
    original_tile_ids = []
    original_tissue_ids = []
    has_tile_id = "tile_id" in tiles_table.columns
    has_tissue_id = "tissue_id" in tiles_table.columns

    for idx, row in tiles_table.iterrows():
        bounds = row.geometry.bounds  # (minx, miny, maxx, maxy)
        minx, miny, maxx, maxy = bounds

        tile_id = row["tile_id"] if has_tile_id else idx
        tissue_id = row["tissue_id"] if has_tissue_id else None

        # Loop order must match timm's patch token order.
        # timm flattens (B, embed, grid_H, grid_W) as flatten(2).transpose(1,2),
        # giving token index k = row * grid_W + col, where row is the y-direction
        # (outer) and col is the x-direction (inner).
        # So j (y) must be the outer loop and i (x) the inner loop.
        for j in range(nh):
            for i in range(nw):
                sub_minx = minx + i * new_width
                sub_miny = miny + j * new_height
                sub_maxx = sub_minx + new_width
                sub_maxy = sub_miny + new_height

                new_tiles.append(box(sub_minx, sub_miny, sub_maxx, sub_maxy))
                if has_tile_id:
                    original_tile_ids.append(tile_id)
                if has_tissue_id:
                    original_tissue_ids.append(tissue_id)

    new_tiles_gdf = gpd.GeoDataFrame(
        {
            "tile_id": np.arange(len(new_tiles)),
            "original_tile_id": original_tile_ids,
            "tissue_id": original_tissue_ids,
        },
        geometry=new_tiles,
    )
    new_tiles_gdf = new_tiles_gdf[
        ["tile_id", "original_tile_id", "tissue_id", "geometry"]
    ]

    # tile_px should be the sub-tile pixel size, not the full parent tile size.
    sub_tile_w = int(tile_spec.width // nw)
    sub_tile_h = int(tile_spec.height // nh)
    new_tile_spec = TileSpec.from_wsidata(
        wsi,
        tile_px=(sub_tile_w, sub_tile_h),
        stride_px=(sub_tile_w, sub_tile_h),
        mpp=tile_spec.mpp,
        tissue_name=tile_spec.tissue_name,
    )

    return new_tiles_gdf, new_tile_spec
