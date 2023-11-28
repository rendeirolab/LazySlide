from pathlib import Path

import openslide
import numpy as np

from .base import ReaderBase


class OpenSlideBackend(ReaderBase):
    """
    Use OpenSlide to interface with image files.

    Depends on `openslide-python <https://openslide.org/api/python/>`_ which wraps the `openslide <https://openslide.org/>`_ C library.

    Args:
        filename (str): path to image file on disk
    """

    def __init__(self, filename):
        self.filename = self.filename
        # self.slide = openslide.open_slide(filename=filename)
        # self.level_count = self.slide.level_count
        self.__level_openslide_handler = {}  # cache level handler
        self._image_array_level = {}  # cache level image in numpy array
        self._openslide_img = 
        self._openslide_fields =
        self.metadata = self.get_metadata()

    def __repr__(self):
        return f"OpenSlideBackend('{self.filename}')"

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
        region = self.slide.read_region(location=(top, left), level=level, size=(width, height))
        region_rgb = pil_to_rgb(region)
        return region_rgb

    def get_level(self, level):
        # return np array as np.uint8

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

        # check if this is fine
        level_shape = self.level_downsample[level]
        level_downsample = self.level_downsample 
        shape = self.dimensions

        filename = os.path.basename(filename)
        metadata = WSIMetaData(
        # file_name=filename, openslide does not provide file_name
        mpp=mpp,
        magnification=mag,
        n_level=n_level,
        shape=shape,
        level_shape=level_shape,
        level_downsample=level_downsample,
        )

        for f in self._get_openslide_field:
            setattr(metadata, f, self._get_openslide_field.get(f))

        return metadata

def pil_to_rgb(image_array_pil):
    """
    Convert PIL RGBA Image to numpy RGB array
    """
    image_array_rgba = np.asarray(image_array_pil)
    image_array = cv2.cvtColor(image_array_rgba, cv2.COLOR_RGBA2RGB).astype(np.uint8)
    return image_array
