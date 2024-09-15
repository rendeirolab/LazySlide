import cv2
import numpy as np

from .base import ScorerBase, ScoreResult
from .utils import dtype_limits


class SplitRGB(ScorerBase):
    """
    Calculate the RGB value of a patch.

    Brightness is calculated as the mean of the pixel values.

    The patch need to be in shape (H, W, 3).

    Parameters
    ----------
    red_threshold : float
        Threshold to determine if a patch is red enough.

    """

    def __init__(
        self,
        threshold: (int, int, int) = (
            0,
            0,
            0,
        ),
        method="mean",
        dim="xyc",
    ):
        self.threshold = np.array(threshold)
        self.method = method
        self.dim = dim
        if dim == "xyc":
            self.func = self._score_xyc
        elif dim == "cyx":
            self.func = self._score_cyx
        else:
            raise ValueError(f"Unknown dim {dim}, should be 'xyc' or 'cyx'")

    def _score_xyc(self, patch, mask=None):
        if mask is not None:
            img = patch[mask]
        else:
            img = patch
        c_int = getattr(img, self.method)(axis=(0, 1))
        return {"red": c_int[0], "green": c_int[1], "blue": c_int[2]}

    def _score_cyx(self, patch, mask=None):
        if mask is not None:
            c_int = [patch[c][mask].mean() for c in range(3)]
        else:
            c_int = [patch[c].mean() for c in range(3)]
        return {"red": c_int[0], "green": c_int[1], "blue": c_int[2]}

    def apply(self, patch, mask=None):
        scores = self.func(patch, mask)
        return ScoreResult(scores=scores, qc=scores > self.threshold)


class Redness(SplitRGB):
    def __init__(self, red_threshold=0.5, **kwargs):
        self.red_threshold = red_threshold
        super().__init__(**kwargs)

    def apply(self, patch, mask=None):
        scores = self.func(patch, mask)
        return ScoreResult(
            scores={"redness": scores["red"]}, qc=scores["red"] > self.red_threshold
        )


class Brightness(ScorerBase):
    def __init__(self, threshold=235):
        self.threshold = threshold

    def apply(self, patch, mask=None) -> ScoreResult:
        if mask is not None:
            bright = patch[mask].mean()
        else:
            bright = patch.mean()
        return ScoreResult(scores={"brightness": bright}, qc=bright < self.threshold)


class Contrast(ScorerBase):
    """
    Calculate the contrast of a patch.

    Contrast is calculated as the standard deviation of the pixel values.

    Parameters
    ----------
    threshold : float
        Threshold to determine if a patch is contrasted or not.
    """

    def __init__(
        self,
        fraction_threshold=0.05,
        lower_percentile=1,
        upper_percentile=99,
    ):
        self.fraction_threshold = fraction_threshold
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

    def apply(self, patch, mask=None):
        patch = np.asarray(patch)
        if patch.dtype == bool:
            ratio = int((patch.max() == 1) and (patch.min() == 0))
        elif patch.ndim == 3:
            if patch.shape[2] == 4:
                patch = cv2.cvtColor(patch, cv2.COLOR_RGBA2RGB)
            if patch.shape[2] == 3:
                patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)

            dlimits = dtype_limits(patch, clip_negative=False)
            limits = np.percentile(
                patch, [self.lower_percentile, self.upper_percentile]
            )
            ratio = (limits[1] - limits[0]) / (dlimits[1] - dlimits[0])
        else:
            raise NotImplementedError("Only support 3D image or 2D image")

        return ScoreResult(
            scores={"contrast": ratio}, qc=ratio > self.fraction_threshold
        )


class Sharpness(ScorerBase):
    """
    Calculate the sharpness of a patch.

    Sharpness is calculated as the variance of the Laplacian of the pixel values.

    Parameters
    ----------
    threshold : float
        Threshold to determine if a patch is sharp or not.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def apply(self, patch, mask=None):
        score = cv2.Laplacian(patch, cv2.CV_64F).var()
        return ScoreResult(scores={"sharpness": score}, qc=score > self.threshold)


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

    def apply(self, patch, mask=None):
        score = cv2.Sobel(patch, 3, 3, 3).var()
        return ScoreResult(scores={"sobel": score}, qc=score > self.threshold)


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

    def apply(self, patch, mask=None):
        score = cv2.Canny(patch, cv2.CV_64F).var()
        return ScoreResult(scores={"canny": score}, qc=score > self.threshold)
