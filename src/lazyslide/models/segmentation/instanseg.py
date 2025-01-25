from __future__ import annotations

import zipfile
from typing import Callable
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import torch
from lazyslide.models.base import SegmentationModel
from .postprocess import cellseg_postprocess
from platformdirs import user_cache_path


def instanseg_preprocess(image: np.ndarray) -> torch.Tensor:
    channel_index = np.argmin(image.shape)
    if channel_index == 0:
        image = image.transpose(1, 2, 0)
    image = image.astype(np.float32).copy()

    for c in range(3):
        min_i, max_i = np.percentile(image[:, :, c], [0.1, 99.9])
        image[:, :, c] = (image[:, :, c] - min_i) / max(1e-3, max_i - min_i)
    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
    return image


class Instanseg(SegmentationModel):
    """Apply the InstaSeg model to the input image."""

    _base_mpp = 0.5
    _fallback_url = "https://github.com/alanocallaghan/instanseg/releases/download/v0.1.0/brightfield_nuclei.zip"
    _source_url = "https://raw.githubusercontent.com/instanseg/instanseg/refs/heads/main/instanseg/bioimageio_models/model-index.json"

    def __init__(self, model_file=None):
        if model_file is None:
            model_url = self._get_wight_url()
            model_file = self.load_weights(model_url)
            with zipfile.ZipFile(model_file, "r") as zip_ref:
                zip_ref.extractall(model_file.parent)
            model_file = model_file.parent / "instanseg.pt"

        self.model = torch.jit.load(model_file)

    def _get_wight_url(self):
        try:
            sources = pd.read_json(self._source_url)
            model_url = sources[sources["name"] == "brightfield_nuclei"].url.values[0]
        except Exception:
            model_url = self._fallback_url
        return model_url

    def get_transform(self):
        return instanseg_preprocess

    def segment(self, image):
        with torch.inference_mode():
            out = self.model(image)
        return out.cpu().numpy().astype(np.int32)

    def get_postprocess_fn(self) -> Callable | None:
        return cellseg_postprocess
