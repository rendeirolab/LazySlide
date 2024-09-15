from __future__ import annotations

import warnings
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
from pathlib import Path
from typing import Callable, Any, Literal

import numpy as np

from lazyslide._const import Key
from wsidata import WSIData
from lazyslide.data.datasets import TileImagesDataset
from lazyslide._utils import default_pbar, chunker, get_torch_device


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


def load_models(
    model_name: str, repo="pytorch/vision", model_path=None, token=None, **kwargs
):
    """Load a model with timm or torch.hub.load"""
    import torch
    from torchvision.models import get_model

    if model_name == "uni":
        from lazyslide.models import UNI

        model = UNI(model_path=model_path, token=token)
    elif model_name == "gigapath":
        from lazyslide.models import GigaPath

        model = GigaPath(model_path=model_path, token=token)
    elif model_name == "conch":
        from lazyslide.models import CONCHVision

        model = CONCHVision(model_path=model_path, token=token)
    elif model_name == "plip":
        from lazyslide.models import PLIPVision

        model = PLIPVision(model_path=model_path, token=token)
    else:
        kwargs = {"weights": "DEFAULT", **kwargs}
        model = get_model(model_name, **kwargs)
    return model, model_name


# TODO: Test if it's possible to load model files
def feature_extraction(
    wsi: WSIData,
    model: str | Any,
    repo: str = "pytorch/vision",
    create_opts: dict = None,
    transform: Callable = None,
    compile: bool = True,
    compile_opts: dict = None,
    device: str = None,
    tile_key: str = Key.tiles,
    feature_key: str = None,
    slide_agg: str | Callable = "mean",
    agg_key: str = "agg",
    batch_size=32,
    num_workers=0,
    mode="batch",  # "batch" or "chunk"
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
    model : str or Any
        The path to the model file or the model object.
    """
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError:
        raise ImportError("Feature extraction requires pytorch and timm (optional).")

    device = device or get_torch_device()

    if isinstance(model, (str, Path)):
        # 1. If model is a path
        model_path = Path(model)
        feature_key = feature_key or model_path.stem
        if model_path.exists():
            try:
                model = torch.load(model)
            except:  # noqa: E722
                model = torch.jit.load(model)
        # 2. If model is plain text
        else:
            create_opts = {} if create_opts is None else create_opts
            model, feature_key = load_models(model, repo=repo, **create_opts)
    elif isinstance(model, Callable):
        model = model
    else:
        raise ValueError(
            "Model must be a model name, "
            "path to the model file, "
            "or a model object."
        )

    if compile:
        try:
            compile_opts = {} if compile_opts is None else compile_opts
            torch.compile(model, **compile_opts)
        except Exception as _:  # noqa: E722
            warnings.warn("Failed to compile the model.", RuntimeWarning)

    try:
        model = model.to(device)
        model.eval()
    except:  # noqa: E722
        pass

    if transform is None:
        transform = get_default_transform()

    if feature_key is None:
        if hasattr(model, "__class__"):
            feature_key = model.__class__.__name__
        elif hasattr(model, "__name__"):
            feature_key = model.__name__
        else:
            feature_key = "features"

    # Create dataloader
    # Auto chunk the wsi tile coordinates to the number of workers'
    tiles_coords = wsi.sdata.shapes[tile_key][["x", "y"]].values
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
                    batch = batch.to(device)
                    output = model(batch)
                    features.append(output.cpu().numpy())
                    progress_bar.update(task, advance=len(batch))
                    del batch  # Free up memory
            # The progress bar may not reach 100% if exit too early
            # Force update
            progress_bar.refresh()
            features = np.vstack(features)

    wsi.add_features(Key.feature(feature_key, tile_key), tile_key, features)
    if return_features:
        return features


def _inference(dataset, chunk, model, queue):
    import torch

    with torch.inference_mode():
        X = []
        for c in chunk:
            img = dataset[c]
            # image to 4d
            img = img.unsqueeze(0)
            output = model(img)
            X.append(output.detach().cpu().numpy())
            queue.put(1)
    return X


def agg_features(
    wsi: WSIData,
    feature_key: str,
    encoder: str | Callable = "mean",
    tile_key: str = Key.tiles,
    by: Literal["slide", "tissue"] = "slide",
    agg_key: str = None,
):
    if agg_key is None:
        agg_key = f"agg_{by}"

    tiles_table = wsi.sdata.shapes[tile_key]
    coords = tiles_table[["x", "y"]]
    feature_key = wsi._check_feature_key(feature_key, tile_key)
    features = wsi.sdata.tables[feature_key].X

    if by == "slide":
        agg_features = _encode_slide(features, encoder, coords)
        agg_features = agg_features.reshape(-1, 1)
    else:
        agg_features = []
        for _, x in tiles_table.groupby("tissue_id"):
            tissue_feature = _encode_slide(
                features[x.index], encoder, coords.iloc[x.index]
            )
            agg_features.append(tissue_feature)
        agg_features = np.vstack(agg_features).T
    wsi.add_agg_features(feature_key, agg_key, agg_features)


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
