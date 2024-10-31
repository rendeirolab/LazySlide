from __future__ import annotations

import warnings
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
from pathlib import Path
from typing import Callable

import numpy as np
from lazyslide._const import Key
from lazyslide._utils import default_pbar, chunker, get_torch_device, find_stack_level
from wsidata import WSIData
from wsidata.io import add_features, add_agg_features


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
        from lazyslide.models import UNI

        model = UNI(model_path=model_path, token=token)
    elif model_name == "gigapath":
        from lazyslide.models import GigaPath

        model = GigaPath(model_path=model_path, token=token)
    elif model_name == "conch":
        from lazyslide.models import CONCH

        model = CONCH(model_path=model_path, token=token)

    elif model_name == "conch_vision":
        from lazyslide.models import CONCHVision

        model = CONCHVision(model_path=model_path, token=token)
    elif model_name == "plip":
        from lazyslide.models import PLIP

        model = PLIP(model_path=model_path, token=token)
    elif model_name == "plip_vision":
        from lazyslide.models import PLIPVision

        model = PLIPVision(model_path=model_path, token=token)
    else:
        from torchvision.models import get_model

        kwargs = {"weights": "DEFAULT", **kwargs}
        model = get_model(model_name, **kwargs)
    return model, model_name


# TODO: Test if it's possible to load model files
def feature_extraction(
    wsi: WSIData,
    model: str | Callable = None,
    model_path: str | Path = None,
    token: str = None,
    load_kws: dict = None,
    transform: Callable = None,
    compile: bool = True,
    compile_kws: dict = None,
    device: str = None,
    tile_key: str = Key.tiles,
    key_added: str = None,
    batch_size: int = 32,
    num_workers: int = 0,
    mode: str = "batch",  # "batch" or "chunk"
    pbar: bool = True,
    return_features: bool = False,
    **kwargs,
):
    """
    Extract features from WSI tiles using a pre-trained model.

    Parameters
    ----------
    wsi : WSI
        The whole-slide image object.
    model : str or model object
        The model used for image feature extraction.
        Built-in foundation models include:

        - 'uni': UNI model from Mahmood Lab.
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
    token : str, optional
        The token for downloading the model from Hugging Face Hub for foundation models.
    load_kws : dict, optional
        Options to pass to the model creation function.
    transform : callable, optional
        The transform function for the input image.
        If not provided, a default ImageNet transform function will be used.
    compile : bool, default: True
        Whether to compile the model for faster inference.
    compile_kws : dict, optional
        Options to pass to the :class:`torch.compile` function.
    device : str, optional
        The device to use for inference. If not provided, the device will be automatically selected.
    tile_key : str, default: 'tiles'
        The key of the tiles dataframe in the spatial data object.
    key_added : str, optional
        The key to store the extracted features.
    batch_size : int, optional
        The batch size for inference.
    num_workers : int, optional
        - mode='batch', The number of workers for data loading.
        - mode='chunk', The number of workers for parallel inference.
    mode : {'batch', 'chunk'}, default: 'batch'
        - 'batch': The data loader will load the data in batches. Only one model instance is launched.
        - 'chunk': Multiple model instances are launched for parallel inference. This mode is only available for CPU.
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

    # If key_added is provided by the user
    user_key = key_added is not None
    load_kws = {} if load_kws is None else load_kws

    if model is not None:
        if isinstance(model, Callable):
            model = model
        elif isinstance(model, str):
            model, model_name = load_models(
                model_name=model, model_path=model_path, token=token, **load_kws
            )
            key_added = key_added or model_name
        else:
            raise ValueError("Model must be a model name or a model object.")
    else:
        if model_path is None:
            raise ValueError("Either model or model_path must be provided.")
        model_path = Path(model_path)
        if model_path.exists():
            try:
                model = torch.load(model_path, **load_kws)
            except:  # noqa: E722
                model = torch.jit.load(model_path, **load_kws)
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")

    # Deal with key_added
    if key_added is None:
        if hasattr(model, "__class__"):
            key_added = model.__class__.__name__
        elif hasattr(model, "__name__"):
            key_added = model.__name__
        else:
            key_added = "features"
    if not user_key:
        key_added = Key.feature(key_added, tile_key)

    if compile:
        try:
            compile_kws = {} if compile_kws is None else compile_kws
            torch.compile(model, **compile_kws)
        except Exception as _:  # noqa: E722
            warnings.warn(
                "Failed to compile the model.",
                RuntimeWarning,
                stacklevel=find_stack_level(),
            )

    try:
        model = model.to(device)
        model.eval()
    except:  # noqa: E722
        pass

    if transform is None:
        transform = get_default_transform()

    # Create dataloader
    # Auto chunk the wsi tile coordinates to the number of workers'
    tiles_coords = wsi.shapes[tile_key][["x", "y"]].values
    n_tiles = len(tiles_coords)

    with default_pbar(disable=not pbar) as progress_bar:
        task = progress_bar.add_task("Extracting features", total=n_tiles)

        if mode == "chunk":
            device = torch.device(device)
            # Check if device is CPU
            if device.type != "cpu":
                raise RuntimeError(
                    "Chunk mode should only used on CPU-based inference."
                )
            if num_workers == 0:
                num_workers = 1

            with Manager() as manager:
                queue = manager.Queue()

                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    wsi.reader.detach_reader()
                    chunks = chunker(np.arange(n_tiles), num_workers)
                    dataset = wsi.ds.tile_images(tile_key=tile_key, transform=transform)
                    futures = [
                        executor.submit(_inference, dataset, chunk, model, queue)
                        for chunk in chunks
                    ]
                    while any(future.running() for future in futures):
                        if queue.empty():
                            continue
                        _ = queue.get()
                        progress_bar.update(task, advance=1)

                    features = []
                    for f in futures:
                        features += f.result()
                    features = np.vstack(features)

        else:
            dataset = wsi.ds.tile_images(tile_key=tile_key, transform=transform)
            loader = DataLoader(
                dataset, batch_size=batch_size, num_workers=num_workers, **kwargs
            )
            # Extract features
            features = []
            with torch.inference_mode():
                for batch in loader:
                    image = batch["image"].to(device)
                    output = model(image)
                    features.append(output.cpu().numpy())
                    progress_bar.update(task, advance=len(image))
                    del batch  # Free up memory
            # The progress bar may not reach 100% if exit too early
            # Force update
            progress_bar.refresh()
            features = np.vstack(features)

    add_features(wsi, key=key_added, tile_key=tile_key, features=features)
    if return_features:
        return features


def _inference(dataset, chunk, model, queue):
    import torch

    with torch.inference_mode():
        X = []
        for c in chunk:
            img = dataset[c]["image"]
            # image to 4d
            img = img.unsqueeze(0)
            output = model(img)
            X.append(output.detach().cpu().numpy())
            queue.put(1)
    return X


def feature_aggregation(
    wsi: WSIData,
    feature_key: str,
    layer_key: str = None,
    encoder: str | Callable = "mean",
    tile_key: str = Key.tiles,
    by: str = "slide",
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
    by : str, default: 'slide'
        The level to aggregate the features.
        - 'slide': Aggregate the features from all tiles in the slide.
        - Column name in tile dataframe: Aggregate the features by specific column.
          For example, to aggregate by tissue pieces, set by='tissue_id'.
    agg_key : str, optional
        The key to store the aggregated features. If not provided, the key will be 'agg_{by}'.

    Returns
    -------
    The aggregated features will be added to the :bdg-danger:`varm` slot of the feature :code:`AnnData`.
    The aggregation operation will be recorded in the :bdg-danger:`uns` slot.

    """
    if agg_key is None:
        agg_key = f"agg_{by}"

    tiles_table = wsi.shapes[tile_key]
    coords = tiles_table[["x", "y"]]
    feature_key = wsi._check_feature_key(feature_key, tile_key)
    if layer_key is None:
        features = wsi.tables[feature_key].X
    else:
        features = wsi.tables[feature_key].layers[layer_key]

    if by == "slide":
        agg_fs = _encode_slide(features, encoder, coords)
        agg_fs = agg_fs.reshape(-1, 1)
    else:
        agg_fs = []
        for _, x in tiles_table.groupby(by):
            tissue_feature = _encode_slide(
                features[x.index], encoder, coords.iloc[x.index]
            )
            agg_fs.append(tissue_feature)
        agg_fs = np.vstack(agg_fs).T
        # The columns of by will also be added to the obs slot
    by_data = tiles_table[by] if by != "slide" else None
    add_agg_features(wsi, feature_key, agg_key, agg_fs, by_key=by, by_data=by_data)


def _encode_slide(features, encoder, coords=None):
    if encoder == "mean":
        agg_features = np.mean(features, axis=0)
    elif encoder == "median":
        agg_features = np.median(features, axis=0)
    elif encoder == "gigapath":
        from lazyslide.models import GigaPathSlideEncoder

        encoder = GigaPathSlideEncoder()
        agg_features = encoder(features, coords)
    elif callable(encoder):
        agg_features = encoder(features)
    else:
        raise ValueError(f"Unknown slide encoding method: {encoder}")

    return agg_features
