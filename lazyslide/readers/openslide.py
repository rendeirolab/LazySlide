from pathlib import Path
from typing import Union

import cv2
import numpy as np

try:
    from openslide import OpenSlide
except Exception as _:
    pass

from .base import ReaderBase, parse_metadata


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
        self.slide = OpenSlide(self.file)
        super().__init__(file, dict(self.slide.properties))
        self._levels = np.arange(self.metadata.n_level)

    def get_patch(
        self,
        left,
        top,
        width,
        height,
        level: int = 0,
        downsample: float = None,
        fill=255,
    ):
        level = self.translate_level(level)
        # TODO: Handle situation that crop outside images
        region = self.slide.read_region(
            location=(top, left), level=level, size=(width, height)
        )
        region_rgb = self._rgba_to_rgb(region)
        return region_rgb

    def get_level(self, level):

        level = self.translate_level(level)

        width, height = self.slide.level_dimensions[level]
        region = self.slide.read_region(
            location=(0, 0), level=level, size=(width, height)
        )
        region_rgb = self._rgba_to_rgb(region)
        return region_rgb
