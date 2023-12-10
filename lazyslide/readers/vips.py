from pathlib import Path
from typing import Union

import cv2
import numpy as np

from .base import ReaderBase, parse_metadata
from .utils import get_crop_left_top_width_height

try:
    import pyvips as vips
except Exception as _:
    pass


VIPS_FORMAT_TO_DTYPE = {
    "uchar": np.uint8,
    "char": np.int8,
    "ushort": np.uint16,
    "short": np.int16,
    "uint": np.uint32,
    "int": np.int32,
    "float": np.float32,
    "double": np.float64,
    "complex": np.complex64,
    "dpcomplex": np.complex128,
}


def vips2numpy(
    vi: "vips.Image",
) -> np.ndarray:
    """Converts a VIPS image into a numpy array"""
    return np.ndarray(
        buffer=vi.write_to_memory(),
        dtype=VIPS_FORMAT_TO_DTYPE[vi.format],
        shape=[vi.height, vi.width, vi.bands],
    )


class VipsReader(ReaderBase):
    def __init__(
        self,
        file: Union[Path, str],
    ):
        self.file = file
        self.__level_vips_handler = {}  # cache level handler

        _vips_img = self._get_vips_level(0)
        _vips_fields = set(_vips_img.get_fields())

        metadata = {}
        for name in _vips_fields:
            metadata[name] = _vips_img.get(name)

        super().__init__(file, metadata)

    def get_patch(
        self,
        left,
        top,
        width,
        height,
        level: int = None,
        downsample: float = None,
        fill=255,
    ):
        """Get a patch by x, y from top-left corner"""
        level = self.translate_level(level)
        img = self._get_vips_level(level)
        patch = self._get_vips_patch(img, left, top, width, height, fill=fill)
        if downsample is not None:
            if downsample != 1:
                patch = patch.resize(1 / downsample)
        patch = vips2numpy(patch)
        return self._rgba_to_rgb(patch)

    def get_level(self, level):
        level = self.translate_level(level)
        img = self._get_vips_level(level)
        img = vips2numpy(img)
        return self._rgba_to_rgb(img)

    def _get_vips_level(self, level=0):
        """Lazy load and load only one for all image level"""
        handler = self.__level_vips_handler.get(level)
        if handler is None:
            handler = vips.Image.new_from_file(str(self.file), fail=True, level=level)
            self.__level_vips_handler[level] = handler
        return handler

    @staticmethod
    def _get_vips_patch(image, left, top, width, height, fill=255):
        bg = [fill]
        crop_left, crop_top, crop_w, crop_h, pos = get_crop_left_top_width_height(
            img_width=image.width,
            img_height=image.height,
            left=left,
            top=top,
            width=width,
            height=height,
        )
        cropped = image.crop(crop_left, crop_top, crop_w, crop_h)
        if pos is None:
            return cropped
        else:
            return cropped.gravity(pos, width, height, background=bg)
