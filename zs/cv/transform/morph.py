import cv2
import numpy as np

from .base import Transform


class MorphOpen(Transform):
    """
    Morphological opening. First applies erosion operation, then dilation.
    Reduces noise by removing small objects from the background.
    Operates on a binary mask.

    Parameters
    ----------
    kernel_size : int
        Size of kernel for default square kernel. Ignored if a custom kernel is specified.
        Defaults to 5.
    n_iterations : int
        Number of opening operations to perform. Defaults to 1.
    """

    def __init__(self, kernel_size=5, n_iterations=1):
        self.set_params(kernel_size=kernel_size, n_iterations=n_iterations)

    def apply(self, mask):
        ksize = self.params["kernel_size"]
        n_iter = self.params["n_iterations"]
        k = np.ones((ksize, ksize), dtype=np.uint8)
        out = cv2.morphologyEx(src=mask, kernel=k, op=cv2.MORPH_OPEN, iterations=n_iter)
        return out


class MorphClose(Transform):
    """
    Morphological closing. First applies dilation operation, then erosion.
    Reduces noise by closing small holes in the foreground.
    Operates on a binary mask.

    Parameters
    ----------
    kernel_size : int
        Size of kernel for default square kernel. Ignored if a custom kernel is specified.
        Defaults to 5.
    n_iterations : int
        Number of opening operations to perform. Defaults to 1.
    """

    def __init__(self, kernel_size=5, n_iterations=1):
        self.set_params(kernel_size=kernel_size, n_iterations=n_iterations)

    def apply(self, mask):
        ksize = self.params["kernel_size"]
        n_iter = self.params["n_iterations"]
        k = np.ones((ksize, ksize), dtype=np.uint8)
        out = cv2.morphologyEx(
            src=mask, kernel=k, op=cv2.MORPH_CLOSE, iterations=n_iter
        )
        return out
