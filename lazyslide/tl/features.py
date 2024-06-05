from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable, Any

import numpy as np

from lazyslide import WSI
from lazyslide.data.datasets import WSIImageDataset


def get_default_transform():
    import torch
    from torchvision.transforms.v2 import Compose, Normalize, ToImage, ToDtype, Resize

    transforms = [
        ToImage(),
        ToDtype(dtype=torch.float32, scale=True),
        Resize(size=(224, 224), antialias=False),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
    # if color_normalize is not None:
    #     pre.append(ColorNormalizer(method=color_normalize))

    return Compose(transforms)


# TODO: Test if it's possible to load model files
def feature_extraction(
    wsi: WSI,
    model: str | Any,
    create_opts: dict = None,
    transform: Callable = None,
    scriptable: bool = True,
    compile: bool = True,
    compile_opts: dict = None,
    device: str = "cpu",
    tile_key: str = "tiles",
    batch_size=32,
    num_workers=0,
    mode="batch",  # "batch" or "chunk"
    **kwargs,
):
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError:
        raise ImportError("Feature extraction requires pytorch and timm (optional).")

    model_path = Path(model)
    model_name = model_path.stem
    if model_path.exists():
        try:
            model = torch.load(model)
        except:  # noqa: E722
            model = torch.jit.load(model)
    else:
        try:
            import timm
        except ImportError:
            raise ImportError("Using model from model market requires timm.")
        try:
            create_opts = {} if create_opts is None else create_opts
            model_name = model
            model = timm.create_model(
                model, pretrained=True, scriptable=scriptable, **create_opts
            )
            if transform is None:
                # data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
                # transform = timm.data.create_transform(**data_cfg)
                transform = get_default_transform()
        except Exception as _:  # noqa: E722
            raise ValueError(f"Model {model} not found.")

    if compile:
        compile_opts = {} if compile_opts is None else compile_opts
        torch.compile(model, **compile_opts)

    model = model.to(device)
    model.eval()

    # Create dataloader
    # Auto chunk the wsi tile coordinates to the number of workers
    if mode == "chunk":
        if num_workers == 0:
            num_workers = 1

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            wsi.reader.detach_reader()
            chunks = chunker(np.arange(len(wsi.sdata.points[tile_key])), num_workers)
            dataset = WSIImageDataset(wsi, transform=transform, key=tile_key)
            futures = [
                executor.submit(_inference, dataset, chunk, model) for chunk in chunks
            ]
            features = []
            for f in futures:
                features += f.result()
            features = np.vstack(features)
    else:
        dataset = WSIImageDataset(wsi, transform=transform, key=tile_key)
        loader = DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, **kwargs
        )
        # Extract features
        features = []
        for batch in loader:
            with torch.inference_mode():
                output = model(batch.to(device))
                features.append(output.cpu().numpy())
        features = np.vstack(features)

    # Write features to WSI
    wsi.add_features(features, tile_key, model_name)


def chunker(seq, num_workers):
    avg = len(seq) / num_workers
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last) : int(last + avg)])
        last += avg

    return out


def _inference(dataset, chunk, model):
    import torch

    with torch.inference_mode():
        X = []
        for c in chunk:
            img = dataset[c]
            # image to 4d
            img = img.unsqueeze(0)
            output = model(img)
            X.append(output.cpu().numpy())
    return X
