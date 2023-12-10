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


def buffer2numpy(
    buffer,
) -> np.ndarray:
    """Converts a VIPS image into a numpy array"""
    return np.ndarray(
        buffer=buffer,
        dtype=VIPS_FORMAT_TO_DTYPE[buffer.format],
        shape=[buffer.height, buffer.width, buffer.bands],
    )


class VipsReader(ReaderBase):
    def __init__(
        self,
        file: Union[Path, str],
        caching: bool = True,
    ):
        self.file = file
        self.caching = caching
        self.__level_vips_handler = {}  # cache level handler
        self.__region_vips_handler = {}  # cache region handler

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
        level: int = 0,
        fill=255,
    ):
        """Get a patch by x, y from top-left corner"""
        level = self.translate_level(level)
        image = self._get_vips_level(level)
        bg = [fill]

        crop_left, crop_top, crop_w, crop_h, pos = get_crop_left_top_width_height(
            img_width=image.width,
            img_height=image.height,
            left=left,
            top=top,
            width=width,
            height=height,
        )
        patch = None
        if self.caching:
            cropped = self.__region_vips_handler[level].fetch(
                crop_left, crop_top, crop_w, crop_h
            )
            if pos is None:
                return np.ndarray(
                    buffer=cropped,
                    dtype=VIPS_FORMAT_TO_DTYPE[image.format],
                    shape=[crop_h, crop_w, image.bands],
                )
            else:
                patch = vips.Image.new_from_buffer(cropped, "").gravity(
                    pos, width, height, background=bg
                )
        else:
            cropped = image.crop(crop_left, crop_top, crop_w, crop_h)
            if pos is None:
                patch = cropped
            else:
                patch = cropped.gravity(pos, width, height, background=bg)

        return vips2numpy(patch)  # self._rgba_to_rgb(patch)

    def get_level(self, level):
        level = self.translate_level(level)
        img = self._get_vips_level(level)
        img = vips2numpy(img)
        return img  # self._rgba_to_rgb(img)

    def _get_vips_level(self, level=0):
        """Lazy load and load only one for all image level"""
        handler = self.__level_vips_handler.get(level)
        if handler is None:
            handler = vips.Image.new_from_file(
                str(self.file), fail=True, level=level, rgb=True
            )
            self.__level_vips_handler[level] = handler
            if self.caching:
                self.__region_vips_handler[level] = vips.Region.new(handler)
        return handler
