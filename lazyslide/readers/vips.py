from typing import Union
from pathlib import Path

import numpy as np

from .base import ReaderBase, WSIMetaData
from .utils import get_crop_xy_wh

try:
    import pyvips as vips
except Exception as e:
    pass

VIPS_FORMAT_TO_DTYPE = {
    'uchar': np.uint8,
    'char': np.int8,
    'ushort': np.uint16,
    'short': np.int16,
    'uint': np.uint32,
    'int': np.int32,
    'float': np.float32,
    'double': np.float64,
    'complex': np.complex64,
    'dpcomplex': np.complex128,
}


def vips2numpy(
        vi: "vips.Image",
) -> np.ndarray:
    """Converts a VIPS image into a numpy array"""
    return np.ndarray(buffer=vi.write_to_memory(),
                      dtype=VIPS_FORMAT_TO_DTYPE[vi.format],
                      shape=[vi.height, vi.width, vi.bands])


class VipsReader(ReaderBase):

    def __init__(self,
                 file: Union[Path, str],
                 ):

        self.file = Path(file)
        self.file_name = file.name
        self.__level_vips_handler = {}
        self._vips_img = self._get_vips_level(0)
        self._vips_fields = set(self._vips_img.get_fields())
        self.metadata = self.get_metadata()

    def get_patch(self,
                  x, y, width, height,
                  level: int = None,
                  downsample: float = None,
                  fill="black",
                  ):
        """Get a patch by x, y from top-left corner"""
        img = self._get_vips_level(level)
        patch = self._get_vips_patch(img, x, y, width, height, fill=fill)
        if downsample is not None:
            if downsample != 1:
                patch = patch.resize(1/downsample)
        return vips2numpy(patch)

    def _get_vips_level(self, level=0):
        """Lazy load and load only one for all image level"""
        handler = self.__level_vips_handler.get(level)
        if handler is not None:
            return handler
        else:
            self.__level_vips_handler[level] = vips.Image.new_from_file(str(self.file), fail=True, level=level)

    @staticmethod
    def _get_vips_patch(image, x, y, width, height, fill="black"):
        bg = [255] if fill == "black" else [0]
        crop_x, crop_y, crop_w, crop_h, pos = get_crop_xy_wh(image.width, image.height, x, y, width, height)
        cropped = image.crop(crop_x, crop_y, crop_w, crop_h)
        if pos is None:
            return cropped
        else:
            return cropped.gravity(pos, width, height, background=bg)

    def _get_vips_field(self, name):
        """Get vips fields safely"""
        if name in self._vips_fields:
            return self._vips_img.get(name)

    def get_metadata(self):

        # search available mpp keys
        mpp_keys = []
        for k in self._vips_fields:
            # Any keys end with .mpp
            if k.lower().endswith(".mpp"):
                mpp_keys.append(k)
        # openslide specific mpp keys
        for k in ('openslide.mpp-x', 'openslide.mpp-y'):
            if k in self._vips_fields:
                mpp_keys.append(k)
        mpp = None
        for k in mpp_keys:
            mpp_tmp = self._get_vips_field(k)
            if mpp_tmp is not None:
                # TODO: Better way to handle this?
                # Current work for 80X
                mpp = np.round(mpp_tmp, decimals=2)
                break

        # search magnification
        mag = self._get_vips_field("openslide.objective-power")
        # TODO: Do we need to handle when level-count is 0?
        n_level = self._get_vips_field("openslide.level-count")

        level_shape = []
        level_downsample = []
        for level in range(n_level):
            height_key = f"openslide.level[{level}].height"
            width_key = f"openslide.level[{level}].width"
            downsample_key = f"openslide.level{level}.downsample"

            height = self._get_vips_field(height_key)
            width = self._get_vips_field(width_key)
            level_shape.append((height, width))

            downsample = self._get_vips_field(downsample_key)
            if downsample is not None:
                downsample = int(downsample)
            level_downsample.append(downsample)

        metadata = WSIMetaData(
            file_name=self.file_name,
            mpp=mpp,
            magnification=mag,
            n_level=n_level,
            shape=level_shape[0],
            level_shape=level_shape,
            level_downsample=level_downsample,
        )

        for f in self._vips_fields:
            setattr(metadata, f, self._vips_img.get(f))

        return metadata
