from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import torch
from wsidata import WSIData
from wsidata.io import add_features, add_agg_features

from lazyslide._const import Key
from lazyslide._utils import default_pbar, get_torch_device
from lazyslide.models import ImageModel


def get_default_transform():
    import torch
    from torchvision.transforms.v2 import Compose, Normalize, ToImage, ToDtype, Resize

    transforms = [
        ToImage(),
        ToDtype(dtype=torch.float32, scale=True),
        Resize(size=(224, 224), antialias=False),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]

    return Compose(transforms)


def load_models(model_name: str, model_path=None, token=None, **kwargs):
    """Load a model with timm or torch.hub.load"""

    if model_name == "uni":
        from lazyslide.models.vision import UNI

        model = UNI(model_path=model_path, token=token)
    elif model_name == "uni2":
        from lazyslide.models.vision import UNI2

        model = UNI2(model_path=model_path, token=token)
    elif model_name == "gigapath":
        from lazyslide.models.vision import GigaPath

        model = GigaPath(model_path=model_path, token=token)
    elif model_name == "conch":
        from lazyslide.models.multimodal import CONCH

        model = CONCH(model_path=model_path, token=token)

    elif model_name == "conch_vision":
        from lazyslide.models.vision import CONCHVision

        model = CONCHVision(model_path=model_path, token=token)
    elif model_name == "plip":
        from lazyslide.models.multimodal import PLIP

        model = PLIP(model_path=model_path, token=token)
    elif model_name == "plip_vision":
        from lazyslide.models.vision import PLIPVision

        model = PLIPVision(model_path=model_path, token=token)
    else:
        from lazyslide.models import TimmModel

        model = TimmModel(model_name, token=token, **kwargs)

    return model, model_name


# TODO: Test if it's possible to load model files
# TODO: Add color normalization
def feature_extraction(
    wsi: WSIData,
    model: str | Callable | ImageModel = None,
    model_path: str | Path = None,
    model_name: str = None,
    jit: bool = False,
    token: str = None,
    load_kws: dict = None,
    transform: Callable = None,
    device: str = None,
    amp: bool = False,
    autocase_dtype: torch.dtype = torch.float16,
    tile_key: str = Key.tiles,
    key_added: str = None,
    batch_size: int = 32,
    num_workers: int = 0,
    pbar: bool = True,
    return_features: bool = False,
    **kwargs,
):
    """
    Extract features from WSI tiles using a pre-trained model.

    Parameters
    ----------
    wsi : :class:`WSIData <wsidata.WSIData>`
        The whole-slide image object.
    model : str or model object
        The model used for image feature extraction.
        Built-in foundation models include:

        - 'uni'/'uni2': UNI model from Mahmood Lab.
        - 'conch': CONCH model from Mahmood Lab for text-image co-embedding.
        - 'conch_vision': CONCH model for only vision task.
        - 'gigapath': GigaPath model from Microsoft.
        - 'plip': PLIP model from Standford Zou Lab for text-image co-embedding.
        - 'plip_vision': PLIP model for only vision task.
        Other models can be loaded from torchvision:
        See https://pytorch.org/vision/stable/models.html for available models.
        Here list some commonly used models in digital pathology:

        - 'resnet50'
        - 'vgg16'
        - 'convnet_base'
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
        The transform function for the input image.
        If not provided, a default ImageNet transform function will be used.
    device : str, optional
        The device to use for inference. If not provided, the device will be automatically selected.
    amp : bool, default: False
        Whether to use automatic mixed precision.
    autocase_dtype : torch.dtype, default: torch.float16
        The dtype for automatic mixed precision.
    tile_key : str, default: 'tiles'
        The key of the tiles dataframe in the spatial data object.
    key_added : str, optional
        The key to store the extracted features.
    batch_size : int, optional
        The batch size for inference.
    num_workers : int, optional
        - mode='batch', The number of workers for data loading.
        - mode='chunk', The number of workers for parallel inference.
    pbar : bool, default: True
        Whether to show progress bar.
    return_features : bool, default: False
        Whether to return the extracted features.

    Returns
    -------
    None or ndarray
        If return_features is True, return the extracted features.

    - The feature matrix will be added to :bdg-danger:`tables` slot of the spatial data object.

    """
    try:
        import torch
        import torchvision
        from torch.utils.data import DataLoader
    except (ImportError, ModuleNotFoundError):
        raise ImportError(
            "Feature extraction requires torch, torchvision and timm (optional)."
        )

    device = device or get_torch_device()

    load_kws = {} if load_kws is None else load_kws

    if model is not None:
        if isinstance(model, Callable):
            model = model
        elif isinstance(model, str):
            model, default_model_name = load_models(
                model_name=model, model_path=model_path, token=token, **load_kws
            )
            if model_name is None:
                model_name = default_model_name
        elif isinstance(model, ImageModel):
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
        elif isinstance(model, ImageModel):
            key_added = model.name
        elif hasattr(model, "__class__"):
            key_added = model.__class__.__name__
        elif hasattr(model, "__name__"):
            key_added = model.__name__
        else:
            key_added = "features"
        key_added = Key.feature(key_added, tile_key)

    try:
        model = model.to(device)
    except:  # noqa: E722
        pass

    if transform is None:
        if isinstance(model, ImageModel):
            transform = model.get_transform()

    # Create dataloader
    # Auto chunk the wsi tile coordinates to the number of workers'
    n_tiles = len(wsi.shapes[tile_key])

    with default_pbar(disable=not pbar) as progress_bar:
        task = progress_bar.add_task("Extracting features", total=n_tiles)

        dataset = wsi.ds.tile_images(tile_key=tile_key, transform=transform)
        loader = DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, **kwargs
        )
        # Extract features
        features = []
        amp_ctx = (
            torch.cuda.amp.autocast(device, autocase_dtype) if amp else nullcontext()
        )
        with amp_ctx, torch.inference_mode():
            for batch in loader:
                image = batch["image"].to(device)
                if isinstance(model, ImageModel):
                    output = model.encode_image(image)
                else:
                    output = model(image)
                if not isinstance(output, np.ndarray):
                    output = output.cpu().numpy()
                features.append(output)
                progress_bar.update(task, advance=len(image))
                del batch  # Free up memory
        # The progress bar may not reach 100% if exit too early
        # Force update
        progress_bar.refresh()
        features = np.vstack(features)

    add_features(wsi, key=key_added, tile_key=tile_key, features=features)
    if return_features:
        return features


def feature_aggregation(
    wsi: WSIData,
    feature_key: str,
    layer_key: str = None,
    encoder: str | Callable = "mean",
    tile_key: str = Key.tiles,
    by: str | Sequence[str] | None = None,
    agg_key: str = None,
):
    """Feature aggregation on different levels.

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
        - 'gigapath': GigaPath slide encoder. The feature must be extracted by GigaPath model.
    tile_key : str, default: 'tiles'
        The key of the tiles dataframe in the spatial data object.
    by : str or array of str, default: None
        The level to aggregate the features.
        - By default will aggregate the features from all tiles in the slide.
        - Column name in tile dataframe: Aggregate the features by specific column.
          For example, to aggregate by tissue pieces, set by='tissue_id'.
    agg_key : str, optional
        The key to store the aggregated features. If not provided, the key will be 'agg_{by}'.

    Returns
    -------
    The aggregated features will be added to the :bdg-danger:`varm` slot of the feature :code:`AnnData`.
    The aggregation operation will be recorded in the :bdg-danger:`uns` slot.

    """
    tiles_table = wsi.shapes[tile_key]
    coords = tiles_table.bounds[["minx", "miny"]]
    feature_key = wsi._check_feature_key(feature_key, tile_key)
    if layer_key is None:
        features = wsi.tables[feature_key].X
    else:
        features = wsi.tables[feature_key].layers[layer_key]

    if by is None:
        if agg_key is None:
            agg_key = "agg_slide"
        agg_fs = _encode_slide(features, encoder, coords)
        agg_fs = agg_fs.reshape(-1, 1)
        agg_annos = {}
    else:
        if isinstance(by, str):
            by = [by]
        if agg_key is None:
            agg_key = f"agg_{'_'.join(by)}"
        agg_fs = []
        agg_annos = []
        for annos, x in tiles_table.groupby(by):
            tissue_feature = _encode_slide(
                features[x.index], encoder, coords.loc[x.index]
            )
            agg_fs.append(tissue_feature)
            agg_annos.append(list(annos))
        agg_fs = np.vstack(agg_fs).T
        agg_annos = {
            "keys": by,
            "values": agg_annos,
        }
        # return agg_fs, agg_annos

        # The columns of by will also be added to the obs slot
    # by_data = tiles_table[by].values if by != "slide" else None
    # add_agg_features(wsi, feature_key, agg_key, agg_fs, by_key=by, by_data=by_data)

    # Add the aggregated features to varm slot of the feature table
    feature_table = wsi.tables[feature_key]
    feature_table.varm[agg_key] = agg_fs

    # Add the aggregation operation to the uns slot
    agg_ops = feature_table.uns.get("agg_ops", {})
    agg_ops[agg_key] = agg_annos
    feature_table.uns["agg_ops"] = agg_ops


def _encode_slide(features, encoder, coords=None):
    if encoder == "mean":
        agg_features = np.mean(features, axis=0)
    elif encoder == "median":
        agg_features = np.median(features, axis=0)
    elif encoder == "gigapath":
        from lazyslide.models.vision import GigaPathSlideEncoder

        encoder = GigaPathSlideEncoder()
        fs = torch.tensor(features).unsqueeze(0)
        cs = torch.tensor(coords.values).unsqueeze(0)
        agg_features = encoder.encode_slide(fs, cs)
    elif callable(encoder):
        agg_features = encoder(features)
    else:
        raise ValueError(f"Unknown slide encoding method: {encoder}")

    return agg_features
