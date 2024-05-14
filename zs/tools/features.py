from __future__ import annotations

from pathlib import Path
from typing import Callable, Any, Literal

from zs import WSI


def feature_extraction(
    wsi: WSI,
    model: str | Any,
    transform: Callable = None,
    scriptable: bool = True,
    compile: bool = True,
    compile_opts: dict = None,
    device: str = "cpu",
    tile_key: str = "tiles",
):
    import timm
    import torch

    if isinstance(model, str):
        try:
            model = timm.create_model(model, pretrained=True, scriptable=scriptable)
            if transform is None:
                data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
                transform = timm.data.create_transform(**data_cfg)
        except Exception as _:  # noqa: E722
            try:
                model = torch.load(model)
            except Exception as _:  # noqa: E722
                model = torch.jit.load(model)

    if compile:
        torch.compile(model, **compile_opts)

    model = model.to(device)
    model.eval()
