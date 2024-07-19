from __future__ import annotations

import warnings
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
from pathlib import Path
from typing import Callable, Any

import numpy as np

from lazyslide import WSI
from lazyslide.data.datasets import TileImagesDataset
from lazyslide.utils import default_pbar, chunker, get_torch_device


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


def load_models(model, repo="pytorch/vision", **kwargs):
    """Load a model with timm or torch.hub.load"""
    import torch

    kwargs = {"weights": "DEFAULT", **kwargs}
    return torch.hub.load(repo, model, **kwargs)


# TODO: Test if it's possible to load model files
def feature_extraction(
    wsi: WSI,
    model: str | Any,
    repo: str = "pytorch/vision",
    create_opts: dict = None,
    transform: Callable = None,
    compile: bool = True,
    compile_opts: dict = None,
    device: str = None,
    tile_key: str = "tiles",
    feature_key: str = None,
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
        model_path = Path(model)
        feature_key = model_path.stem
        if model_path.exists():
            try:
                model = torch.load(model)
            except:  # noqa: E722
                model = torch.jit.load(model)
        else:
            create_opts = {} if create_opts is None else create_opts
            model = load_models(model, repo=repo, **create_opts)
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
    tiles_count = len(wsi.sdata.points[tile_key])

    with default_pbar(disable=not pbar) as progress_bar:
        task = progress_bar.add_task("Extracting features", total=tiles_count)

        if mode == "chunk":
            if num_workers == 0:
                num_workers = 1

            with Manager() as manager:
                queue = manager.Queue()

                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    wsi.reader.detach_reader()
                    chunks = chunker(np.arange(tiles_count), num_workers)
                    dataset = TileImagesDataset(wsi, transform=transform, key=tile_key)
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
            dataset = TileImagesDataset(wsi, transform=transform, key=tile_key)
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
            # The progress bar may not reach 100% if exit too early
            # Force update
            progress_bar.refresh()
            features = np.vstack(features)

    # Write features to WSI
    wsi.add_features(features, tile_key, feature_key)
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
            X.append(output.cpu().numpy())
            queue.put(1)
    return X
