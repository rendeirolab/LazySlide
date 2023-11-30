from pathlib import Path
from typing import Union

import numpy as np
import cv2
from openslide import OpenSlide

from .base import ReaderBase, WSIMetaData


class OpenSlideReader(ReaderBase):
    """
    Use OpenSlide to interface with image files.

    Depends on `openslide-python <https://openslide.org/api/python/>`_ which wraps the `openslide <https://openslide.org/>`_ C library.

    Args:
        filename (str): path to image file on disk
    """

    def __init__(
        self,
        file: Union[Path, str],
    ):
        super().__init__(file)
        self.filename = self.file.name
        self.slide = OpenSlide(filename=self.file)
        self.properties = self.slide.properties
        # self.level_count = self.slide.level_count
        self.metadata = self.get_metadata()

    def __repr__(self):
        return f"OpenSlideReader('{self.filename}')"

    def get_patch(
        self,
        left,
        top,
        width,
        height,
        level: int = None,
        downsample: float = None,
        fill="black",
    ):
        # TODO: Handle situation that crop outside images
        region = self.slide.read_region(
            location=(top, left), level=level, size=(width, height)
        )
        region_rgb = pil_to_rgb(region)
        return region_rgb

    def get_level(self, level):
        # return np array as np.uint8
        if level + 1 > self.metadata.n_level:
            raise ValueError(f"Requested level {level} is not available")

        width, height = self.slide.level_dimensions[level]
        region = self.slide.read_region(
            location=(0, 0), level=level, size=(width, height)
        )
        region_rgb = pil_to_rgb(region)
        return region_rgb

    def _get_openslide_field(self, name):
        """Get vips fields safely"""
        if name in self.properties:
            return self.properties.get(name)

    def get_metadata(self):
        # search available mpp keys
        mpp_keys = []
        for k in self.properties:
            # Any keys end with .mpp
            if k.lower().endswith(".mpp"):
                mpp_keys.append(k)
        # openslide specific mpp keys
        for k in ("openslide.mpp-x", "openslide.mpp-y"):
            if k in self.properties:
                mpp_keys.append(k)
        mpp = None
        for k in mpp_keys:
            mpp_tmp = float(self._get_openslide_field(k))
            if mpp_tmp is not None:
                # TODO: Better way to handle this?
                # Current work for 80X
                mpp = np.round(mpp_tmp, decimals=2)
                break

        # search magnification
        mag = self._get_openslide_field("openslide.objective-power")
        # TODO: Do we need to handle when level-count is 0?
        n_level = int(self._get_openslide_field("openslide.level-count"))

        # check if this works
        level_shape = self.slide.level_dimensions
        level_downsample = self.slide.level_downsamples
        shape = self.slide.dimensions

        metadata = WSIMetaData(
            file_name=self.filename,
            mpp=mpp,
            magnification=mag,
            n_level=n_level,
            shape=shape,
            level_shape=level_shape,
            level_downsample=level_downsample,
        )

        for f in self.properties:
            setattr(metadata, f, self.properties.get(f))

        return metadata


def pil_to_rgb(image_array_pil):
    """
    Convert PIL RGBA Image to numpy RGB array
    """
    image_array_rgba = np.asarray(image_array_pil)
    image_array = cv2.cvtColor(image_array_rgba, cv2.COLOR_RGBA2RGB).astype(np.uint8)
    return image_array
