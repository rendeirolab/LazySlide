from __future__ import annotations

from functools import singledispatch
from typing import Optional, List, Mapping

import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel

# Tuple is not serializable for anndata
SHAPE = List[int]


class SlideMetadata(BaseModel):
    mpp: Optional[float] = None
    magnification: Optional[float] = None
    shape: SHAPE
    n_level: int
    level_shape: List[SHAPE]
    level_downsample: List[float]

    @classmethod
    def from_mapping(self, metadata: Mapping):
        metadata = parse_metadata(metadata)
        return SlideMetadata(**metadata)

    def _repr_html_(self):
        return (
            "<h4>Slide Metadata</h4><table><tr><th>Field</th><th>Value</th></tr>"
            + "".join(
                f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in self.dict().items()
            )
            + "</table>"
        )


class ReaderBase:
    file: str
    metadata: SlideMetadata
    name = "base"
    _reader = None

    def __repr__(self):
        return f"{self.__class__.__name__}('{self.file}')"

    def translate_level(self, level):
        levels = np.arange(self.metadata.n_level)
        if level >= len(levels):
            raise ValueError(f"Request level {level} not exist")
        return levels[level]

    def get_region(self, x, y, width, height, level=0, **kwargs):
        """Get a region from image with top-left corner
        This should return a numpy array in xyc format
        """
        raise NotImplementedError

    def get_center(self, x, y, width, height, level=0, **kwargs):
        """Get a patch from image with center"""
        x -= width / 2
        y -= height / 2
        return self.get_region(x, y, width, height, level=level, **kwargs)

    def get_level(self, level):
        """Get the image level in numpy array"""
        raise NotImplementedError

    def set_metadata(self, metadata: SlideMetadata | Mapping):
        if isinstance(metadata, SlideMetadata):
            self.metadata = metadata
        else:
            self.metadata = SlideMetadata.from_mapping(metadata)

    def detach_reader(self):
        # The basic fallback implementation to detach reader
        # In real implementation, this should close the file handler
        raise NotImplementedError

    @staticmethod
    def resize_img(
        img: np.ndarray,
        scale: float,
    ):
        dim = np.asarray(img.shape)
        dim = np.array(dim * scale, dtype=int)
        return cv2.resize(img, dim)


@singledispatch
def convert_image(img):
    raise NotImplementedError(f"Unsupported type {type(img)}")


@convert_image.register(Image.Image)
def _(img: Image.Image):
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGBA2RGB).astype(np.uint8)


MAG_KEY = "objective-power"
MPP_KEYS = ("mpp-x", "mpp-y")

LEVEL_HEIGHT_KEY = lambda level: f"level[{level}].height"  # noqa: E731
LEVEL_WIDTH_KEY = lambda level: f"level[{level}].width"  # noqa: E731
LEVEL_DOWNSAMPLE_KEY = lambda level: f"level[{level}].downsample"  # noqa: E731


def parse_metadata(metadata: Mapping):
    metadata = dict(metadata)
    new_metadata = {}
    for k, v in metadata.items():
        new_metadata[".".join(k.split(".")[1::])] = v
    metadata.update(new_metadata)

    fields = set(metadata.keys())

    mpp_keys = []
    # openslide specific mpp keys
    if MPP_KEYS[0] in fields:
        mpp_keys.append(MPP_KEYS[0])
    if MPP_KEYS[1] in fields:
        mpp_keys.append(MPP_KEYS[1])
    for k in fields:
        # Any keys end with .mpp
        if k.lower().endswith("mpp"):
            mpp_keys.append(k)

    mpp = None
    for k in mpp_keys:
        mpp_tmp = metadata.get(k)
        if mpp_tmp is not None:
            mpp = float(mpp_tmp)

    # search magnification
    # search other available mpp keys
    mag_keys = []
    if MAG_KEY in fields:
        mag_keys.append(MAG_KEY)
    for k in fields:
        # Any keys end with .mpp
        if k.lower().endswith("appmag"):
            mag_keys.append(k)

    mag = None
    for k in mag_keys:
        mag_tmp = metadata.get(k)
        if mag_tmp is not None:
            mag = float(mag_tmp)

    level_shape = []
    level_downsample = []

    # Get the number of levels
    n_level = 0
    while f"level[{n_level}].width" in fields:
        n_level += 1

    for level in range(n_level):
        height = metadata.get(LEVEL_HEIGHT_KEY(level))
        width = metadata.get(LEVEL_WIDTH_KEY(level))
        downsample = metadata.get(LEVEL_DOWNSAMPLE_KEY(level))

        level_shape.append((int(height), int(width)))

        if downsample is not None:
            downsample = float(downsample)
        level_downsample.append(downsample)

    shape = level_shape[0]

    metadata = {
        "mpp": mpp,
        "magnification": mag,
        "shape": shape,
        "n_level": n_level,
        "level_shape": level_shape,
        "level_downsample": level_downsample,
    }

    return metadata
