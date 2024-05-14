import cv2
import numpy as np

from .base import Transform


class BinaryThreshold(Transform):
    """
    Binary thresholding transform to create a binary mask.
    If input image is RGB it is first converted to greyscale, otherwise the input must have 1 channel.

    Parameters
    ----------
    use_otsu : bool
        Whether to use Otsu's method to automatically determine optimal threshold. Defaults to True.
    threshold : int
        Specified threshold. Ignored if ``use_otsu is True``. Defaults to 0.
    inverse : bool
        Whether to use inverse threshold. If using inverse threshold, pixels below the threshold will
        be returned as 1. Otherwise, pixels below the threshold will be returned as 0. Defaults to ``False``.


    """

    def __init__(self, use_otsu=True, threshold=0, inverse=False):
        self.type = cv2.THRESH_BINARY_INV if inverse else cv2.THRESH_BINARY
        if use_otsu:
            self.type += cv2.THRESH_OTSU
        self.set_params(use_otsu=use_otsu, threshold=threshold, inverse=inverse)

    def apply(self, image):
        if image.ndim > 2:
            raise ValueError("Must be greyscale image (H, W) or binary mask (H, W).")

        _, out = cv2.threshold(
            src=image,
            thresh=self.params["threshold"],
            maxval=255,
            type=self.type,
        )
        if self.params["inverse"]:
            out = 1 - out
        return out


class ArtifactFilterThreshold(Transform):
    """
    Artifact filter thresholding transform to create a binary mask.

    Parameters
    ----------
    threshold : int
        Threshold value for artifact filter.

    """

    def __init__(self, threshold=0):
        self.set_params(threshold=threshold)

    def apply(self, image):
        red_channel = image[:, :, 0].astype(float)
        green_channel = image[:, :, 1].astype(float)
        blue_channel = image[:, :, 2].astype(float)

        red_to_green_mask = np.maximum(red_channel - green_channel, 0)
        blue_to_green_mask = np.maximum(blue_channel - green_channel, 0)

        tissue_heatmap = red_to_green_mask * blue_to_green_mask

        _, out = cv2.threshold(
            src=tissue_heatmap.astype(np.uint8),
            thresh=self.params["threshold"],
            maxval=255,
            type=(cv2.THRESH_BINARY + cv2.THRESH_OTSU),
        )
        return out
