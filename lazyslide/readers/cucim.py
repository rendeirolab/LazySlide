from pathlib import Path
from typing import Union

import numpy as np


try:
    from cucim import CuImage
    from skimage.util import img_as_float32
except Exception as _:
    pass

from .base import ReaderBase, WSIMetaData


def cucim2numpy(img: "CuImage") -> np.ndarray:
    return ((img_as_float32(np.asarray(img))) * 255).astype(np.uint8)


class CuCIMReader(ReaderBase):
    def __init__(
        self,
        file: Union[Path, str],
        raw_metadata: bool = False,
        device=None,
    ):
        self.slide = CuImage(str(file))

        level_info = self.slide.resolutions
        n_level = level_info["level_count"]
        level_shape = level_info["level_dimensions"]
        level_downsample = level_info["level_downsamples"]
        shape = self.slide.shape[0:2]

        mpp = None
        magnification = None

        raw = {}
        for field, info in self.slide.metadata.items():
            for prop_k, prop_v in info.items():
                if prop_k.lower().endswith("mpp"):
                    mpp = float(prop_v)
                elif prop_k.lower().endswith("appmag"):
                    magnification = float(prop_v)
                raw[prop_k] = prop_v

        metadata = WSIMetaData(
            filename=file,
            mpp=mpp,
            magnification=magnification,
            shape=shape,
            n_level=n_level,
            level_shape=level_shape,
            level_downsample=level_downsample,
        )

        super().__init__(file, metadata)

    def get_patch(self, left, top, width, height, level=0, **kwargs):
        patch = self.slide.read_region(
            (left, top),
            (height, width),
            level=level,
        )
        return cucim2numpy(patch)

    def get_level(self, level):
        level_img = self.slide.read_region(level=level)
        return cucim2numpy(level_img)
