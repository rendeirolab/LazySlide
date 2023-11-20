from dataclasses import dataclass, field
from typing import Dict

import numpy as np
from skimage.transform import resize


class BackendBase:

    def get_patch(self):
        """Get a patch from image with top-left corner"""
        raise NotImplemented

    def get_cell(self):
        """Get a patch from image with center"""
        raise NotImplemented

    def get_metadata(self):
        raise NotImplemented

    @staticmethod
    def resize_img(img: np.ndarray,
                   output_shape,
                   ):
        return resize(
            img,
            output_shape,
        )


@dataclass
class WSIMetaData:
    file_id: str
    mpp: field(None)
    image_shape: Dict

    def get_image_shape(self):
        pass
