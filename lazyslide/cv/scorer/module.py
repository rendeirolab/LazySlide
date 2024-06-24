import cv2
import numpy as np

from .base import ScorerBase
from .utils import dtype_limits


class Brightness(ScorerBase):
    """
    Calculate the brightness of a patch.

    Brightness is calculated as the mean of the pixel values.

    Parameters
    ----------
    threshold : float
        Threshold to determine if a patch is bright or not.
    """

    name = "brightness"

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def get_score(self, patch) -> float:
        return patch.mean()

    def filter(self, scores):
        return scores > self.threshold


class Contrast(ScorerBase):
    """
    Calculate the contrast of a patch.

    Contrast is calculated as the standard deviation of the pixel values.

    Parameters
    ----------
    threshold : float
        Threshold to determine if a patch is contrasted or not.
    """

    name = "contrast"

    def __init__(
        self,
        fraction_threshold=0.05,
        lower_percentile=1,
        upper_percentile=99,
    ):
        self.fraction_threshold = fraction_threshold
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

    def get_score(self, patch) -> float:
        patch = np.asarray(patch)
        if patch.dtype == bool:
            return int((patch.max() == 1) and (patch.min() == 0))
        if patch.ndim == 3:
            if patch.shape[2] == 4:
                patch = cv2.cvtColor(patch, cv2.COLOR_RGBA2RGB)
            if patch.shape[2] == 3:
                patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)

            dlimits = dtype_limits(patch, clip_negative=False)
            limits = np.percentile(
                patch, [self.lower_percentile, self.upper_percentile]
            )
            ratio = (limits[1] - limits[0]) / (dlimits[1] - dlimits[0])

            return ratio

    def filter(self, scores):
        return scores > self.fraction_threshold


class Sharpness(ScorerBase):
    """
    Calculate the sharpness of a patch.

    Sharpness is calculated as the variance of the Laplacian of the pixel values.

    Parameters
    ----------
    threshold : float
        Threshold to determine if a patch is sharp or not.
    """

    name = "sharpness"

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def get_score(self, patch) -> float:
        return cv2.Laplacian(patch, cv2.CV_64F).var()

    def filter(self, scores):
        return scores > self.threshold


class Sobel(ScorerBase):
    """
    Calculate the sobel of a patch.

    Sobel is calculated as the variance of the Sobel of the pixel values.

    Parameters
    ----------
    threshold : float
        Threshold to determine if a patch is sharp or not.
    """

    name = "sobel"

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def get_score(self, patch) -> float:
        return cv2.Sobel(patch, 3, 3).var()


class Canny(ScorerBase):
    """
    Calculate the canny of a patch.

    Canny is calculated as the variance of the Canny of the pixel values.

    Parameters
    ----------
    threshold : float
        Threshold to determine if a patch is sharp or not.
    """

    name = "canny"

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def get_score(self, patch) -> float:
        return cv2.Canny(patch, cv2.CV_64F).var()
