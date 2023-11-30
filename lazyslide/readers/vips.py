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
        super().__init__(file)
        self.__level_vips_handler = {}  # cache level handler
        self._image_array_level = {}  # cache level image in numpy array
        self._vips_img = self._get_vips_level(0)
        self._vips_fields = set(self._vips_img.get_fields())
        self.metadata = self.get_metadata()

    def get_patch(self,
                  left, top, width, height,
                  level: int = None,
                  downsample: float = None,
                  fill="black",
                  ):
        """Get a patch by x, y from top-left corner"""
        img = self._get_vips_level(level)
        patch = self._get_vips_patch(img, left, top, width, height, fill=fill)
        if downsample is not None:
            if downsample != 1:
                patch = patch.resize(1 / downsample)
        img_arr = vips2numpy(patch)
        return cv2.cvtColor(img_arr, cv2.COLOR_RGBA2RGB).astype(np.uint8)

    def get_level(self, level):
        img_arr = self._image_array_level.get(level)
        if img_arr is None:
            img = self._get_vips_level(level)
            img_arr = vips2numpy(img)
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGBA2RGB).astype(np.uint8)
            self._image_array_level[level] = img_arr
        return img_arr

    def _get_vips_level(self, level=0):
        """Lazy load and load only one for all image level"""
        handler = self.__level_vips_handler.get(level)
        if handler is None:
            handler = vips.Image.new_from_file(
                str(self.file), fail=True, level=level)
            self.__level_vips_handler[level] = handler
        return handler

    @staticmethod
    def _get_vips_patch(image, left, top, width, height, fill="white"):
        bg = [255] if fill == "black" else [0]
        crop_left, crop_top, crop_w, crop_h, pos = \
            get_crop_left_top_width_height(
                img_width=image.width, img_height=image.height,
                left=left, top=top, width=width, height=height)
        cropped = image.crop(crop_left, crop_top, crop_w, crop_h)
        if pos is None:
            return cropped
        else:
            return cropped.gravity(pos, width, height, background=bg)

    def get_metadata(self):
        metadata = {}
        for name in self._vips_fields:
            metadata[name] = self._vips_img.get(name)

        return parse_metadata(self.filename, metadata)
