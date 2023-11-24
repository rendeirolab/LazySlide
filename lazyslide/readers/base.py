from dataclasses import dataclass, field
from typing import List

import cv2
import numpy as np


@dataclass
class WSIMetaData:
    file_name: str
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

    def get_patch(self, x, y, width, height, level=0, **kwargs):
        """Get a patch from image with top-left corner"""

        raise NotImplemented

    def get_center(self, x, y, width, height, level=0, **kwargs):
        """Get a patch from image with center"""
        x -= width / 2
        y -= height / 2
        return self.get_patch(x, y, width, height, level=level, **kwargs)

    def get_metadata(self):
        raise NotImplemented

    @staticmethod
    def resize_img(img: np.ndarray,
                   scale: float,
                   ):
        dim = np.asarray(img.shape)
        dim = np.array(dim * scale, dtype=int)
        return cv2.resize(img, dim)

    def _get_ops_params(self,
                        level: int = None,
                        mpp: float = None,
                        magnification: int = None, ):

        has_level = level is not None
        has_mpp = mpp is not None
        has_mag = magnification is not None

        # only one argument is allowed
        if np.sum([has_level, has_mpp, has_mag]) != 1:
            raise ValueError("Please specific one and only one argument,"
                             "level, mpp or magnification")

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
            raise ValueError(f"Downsample factor must >= 1, "
                             f"the input is {factor}")

        level_downsample = np.array(self.metadata.level_downsample)
        # Get levels that can be downsample
        avail_downsample = level_downsample[level_downsample < factor]

        if len(avail_downsample) == 0:
            return
        use_downsample = avail_downsample[np.argmin(np.abs(avail_downsample - factor))]
        return np.where(level_downsample == use_downsample)[0][0]

