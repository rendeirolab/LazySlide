from pathlib import Path
from typing import Union

import cv2
import numpy as np
from openslide import OpenSlide

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
        super().__init__(file)
        self.slide = OpenSlide(self.file)
        self.metadata = self.get_metadata()

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

    def get_metadata(self):
        return parse_metadata(self.filename, dict(self.slide.properties))


def pil_to_rgb(image_array_pil):
    """
    Convert PIL RGBA Image to numpy RGB array
    """
    image_array_rgba = np.asarray(image_array_pil)
    image_array = cv2.cvtColor(image_array_rgba, cv2.COLOR_RGBA2RGB).astype(np.uint8)
    return image_array
