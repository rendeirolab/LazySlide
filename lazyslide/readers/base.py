from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union, Dict

import cv2
import numpy as np


@dataclass
class WSIMetaData:
    filename: str
    mpp: field(default=None)
    magnification: field(default=None)
    shape: tuple
    n_level: int
    level_shape: List
    level_downsample: List

    def get_image_shape(self):
        pass


class ReaderBase:
    metadata: WSIMetaData

    def __init__(
        self,
        file: Union[Path, str],
        metadata: Union[Dict, WSIMetaData],
        raw_metadata: bool = False,
    ):
        self.file = Path(file)
        self.filename = self.file.name
        if isinstance(metadata, WSIMetaData):
            self.metadata = metadata
        else:
            self.metadata = parse_metadata(
                self.filename, metadata, attach_raw=raw_metadata
            )
        self._levels = np.arange(self.metadata.n_level)

    def __repr__(self):
        return f"{self.__class__.__name__}('{self.filename}')"

    def translate_level(self, level):
        return self._levels[level]

    def get_patch(self, left, top, width, height, level=0, **kwargs):
        """Get a patch from image with top-left corner"""
        raise NotImplementedError

    def get_center(self, left, top, width, height, level=0, **kwargs):
        """Get a patch from image with center"""
        left -= width / 2
        top -= height / 2
        return self.get_patch(left, top, width, height, level=level, **kwargs)

    def get_level(self, level):
        """Get the image level in numpy array"""
        raise NotImplementedError

    def get_metadata(self):
        return self.metadata

    def detach_handler(self):
        return NotImplementedError

    def attach_handler(self):
        return NotImplementedError

    @staticmethod
    def resize_img(
        img: np.ndarray,
        scale: float,
    ):
        dim = np.asarray(img.shape)
        dim = np.array(dim * scale, dtype=int)
        return cv2.resize(img, dim)

    @staticmethod
    def _rgba_to_rgb(img):
        image_array_rgba = np.asarray(img)
        image_array = cv2.cvtColor(image_array_rgba, cv2.COLOR_RGBA2RGB).astype(
            np.uint8
        )
        return image_array

    def _get_ops_params(
        self,
        level: int = None,
        mpp: float = None,
        magnification: int = None,
    ):
        has_level = level is not None
        has_mpp = mpp is not None
        has_mag = magnification is not None

        # only one argument is allowed
        if np.sum([has_level, has_mpp, has_mag]) != 1:
            raise ValueError(
                "Please specific one and only one argument,"
                "level, mpp or magnification"
            )

        ops_resize = False
        ops_level = 0
        ops_factor = 1
        if has_level:
            if level >= self.metadata.n_level:
                raise ValueError(f"Cannot operate on non-exist level={level}")
            ops_level = level
        else:
            level_downsample = np.array(self.metadata.level_downsample)
            if has_mpp:
                request_factor = mpp / self.metadata.mpp
            else:
                request_factor = self.metadata.magnification / magnification
            if request_factor < 1:
                ops_resize = True
                ops_factor = request_factor
            elif request_factor > 1:
                level = np.where(level_downsample == request_factor)
                # If no match level
                if len(level) == 0:
                    ops_resize = True
                    ops_level = self._get_best_level_to_downsample(request_factor)
                    # This factor is corresponding to the level that it works on
                    ops_factor = request_factor / level_downsample[ops_level]
                else:
                    ops_level = level[0][0]

        return ops_resize, ops_level, ops_factor

    def _get_best_level_to_downsample(self, factor):
        if factor <= 1:
            raise ValueError(f"Downsample factor must >= 1, " f"the input is {factor}")

        level_downsample = np.array(self.metadata.level_downsample)
        # Get levels that can be downsample
        avail_downsample = level_downsample[level_downsample < factor]

        if len(avail_downsample) == 0:
            return
        use_downsample = avail_downsample[np.argmin(np.abs(avail_downsample - factor))]
        return np.where(level_downsample == use_downsample)[0][0]


MAG_KEY = "openslide.objective-power"
MPP_KEYS = ("openslide.mpp-x", "openslide.mpp-y")
N_LEVEL_KEY = "openslide.level-count"

LEVEL_HEIGHT_KEY = lambda level: f"openslide.level[{level}].height"
LEVEL_WIDTH_KEY = lambda level: f"openslide.level[{level}].width"
LEVEL_DOWNSAMPLE_KEY = lambda level: f"openslide.level[{level}].downsample"


def parse_metadata(filename, metadata: Dict, attach_raw=False):
    fields = set(metadata.keys())

    mpp_keys = []
    # openslide specific mpp keys
    for k in MPP_KEYS:
        if k in fields:
            mpp_keys.append(k)
    # search other available mpp keys
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
    mag_keys = []
    if MAG_KEY in fields:
        mag_keys.append(MAG_KEY)

    # search other available mpp keys
    for k in fields:
        # Any keys end with .mpp
        if k.lower().endswith("appmag"):
            mag_keys.append(k)

    mag = None
    for k in mag_keys:
        mag_tmp = metadata.get(k)
        if mag_tmp is not None:
            mag = float(mag_tmp)

    # TODO: Do we need to handle when level-count is 0?
    n_level = 1
    level_shape = []
    level_downsample = []
    shape = (None, None)

    if N_LEVEL_KEY in fields:
        n_level_tmp = metadata.get(N_LEVEL_KEY)
        if n_level_tmp is not None:
            n_level = int(n_level_tmp)

        for level in range(n_level):
            height = metadata.get(LEVEL_HEIGHT_KEY(level))
            width = metadata.get(LEVEL_WIDTH_KEY(level))
            downsample = metadata.get(LEVEL_DOWNSAMPLE_KEY(level))

            level_shape.append((int(height), int(width)))

            if downsample is not None:
                downsample = float(downsample)
            level_downsample.append(downsample)

        shape = level_shape[0]

    wsi_meta = WSIMetaData(
        filename=filename,
        mpp=mpp,
        magnification=mag,
        shape=shape,
        n_level=n_level,
        level_shape=level_shape,
        level_downsample=level_downsample,
    )
    if attach_raw:
        for k, v in metadata.items():
            setattr(wsi_meta, k, v)

    return wsi_meta
