import cv2
import numpy as np

from .base import Transform


class MedianBlur(Transform):
    """
    Median blur kernel.

    Parameters
    ----------
    kernel_size : int
            Width of kernel. Must be an odd number. Defaults to 5.

    """

    def __init__(self, kernel_size=5):
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd number")
        self.set_params(kernel_size=kernel_size)

    def apply(self, image):
        return cv2.medianBlur(image, ksize=self.params["kernel_size"])


class GaussianBlur(Transform):
    """
    Gaussian blur kernel.

    Parameters
    ----------
    kernel_size : int
            Width of kernel. Must be an odd number. Defaults to 5.
    sigma : float
            Variance of Gaussian kernel. Variance is assumed to be equal in X and Y axes. Defaults to 5.

    """

    def __init__(self, kernel_size=5, sigma=5):
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd number")
        self.set_params(kernel_size=kernel_size, sigma=sigma)

    def apply(self, image):
        k_size = self.params["kernel_size"]
        sigma = self.params["sigma"]
        out = cv2.GaussianBlur(
            image,
            ksize=(k_size, k_size),
            sigmaX=sigma,
            sigmaY=sigma,
        )
        return out


class BoxBlur(Transform):
    """
    Box (average) blur kernel.

    Parameters
    ----------
    kernel_size : int
        Width of kernel. Defaults to 5.
    """

    def __init__(self, kernel_size=5):
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd number")
        self.set_params(kernel_size=kernel_size)

    def apply(self, image):
        ksize = self.params["kernel_size"]
        return cv2.boxFilter(image, ksize=(ksize, ksize), ddepth=-1)
