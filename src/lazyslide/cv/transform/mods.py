from collections import namedtuple

import cv2
import numpy as np

# Base class


class Transform:
    """Image Transform base class.

    Image -> Image
    """

    params: dict = {}

    def __repr__(self):
        # print params
        params_str = ", ".join([f"{k}={v}" for k, v in self.params.items()])
        return f"{self.__class__.__name__}({params_str})"

    def __call__(self, image):
        if isinstance(image, np.ndarray):
            image = image.astype(np.uint8)
            processed_image = self.apply(image)
            return processed_image.astype(np.uint8)
        else:
            raise TypeError(f"Input must be np.ndarray, got {type(image)}")

    def apply(self, image):
        """Perform transformation"""
        raise NotImplementedError

    def set_params(self, **params):
        self.params.update(params)
        for k, v in params.items():
            if hasattr(self, k):
                raise ValueError(
                    f"Parameter {k} is not valid for {self.__class__.__name__}"
                )
            setattr(self, k, v)


class Compose(Transform):
    """Compose multiple transforms together."""

    def __init__(self, transforms):
        self.pipeline = transforms

    def apply(self, image):
        for p in self.pipeline:
            image = p(image)
        return image


# ================= Blurry Modules =================


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


# ================= Thresholding Modules =================


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


# ================= Morphological Modules =================


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


# ================= Foreground Detection Modules =================


TissueInstance = namedtuple("TissueInstance", ["id", "contour", "holes"])


class ForegroundDetection(Transform):
    """
    Foreground detection for binary masks. Identifies regions that have a total area greater than
    specified threshold. Supports including holes within foreground regions, or excluding holes
    above a specified area threshold.

    Parameters
    ----------
    min_foreground_area : int or within (0, 1)
        If int, minimum area of detected foreground regions, in pixels area.
        If within (0, 1), minimum area of detected foreground regions, in proportion to the total image area.
    min_hole_area : int or within (0, 1)
        If int, maximum size of allowed holes in foreground regions, in pixels area.
        If within (0, 1), maximum size of allowed holes in foreground regions, in proportion to the total image area.
    detect_holes : bool
        Whether to detect holes in foreground regions. Defaults to True.

    """

    def __init__(
        self,
        min_foreground_area=0.01,
        min_hole_area=0.0001,
        detect_holes=True,
    ):
        self.set_params(
            min_foreground_area=min_foreground_area,
            min_hole_area=min_hole_area,
            detect_holes=detect_holes,
        )

    def __call__(self, mask):
        if isinstance(mask, np.ndarray):
            mask = mask.astype(np.uint8)
        return self.apply(mask)

    def apply(self, mask):
        detect_holes = self.params["detect_holes"]
        min_foreground_area = self.params["min_foreground_area"]
        min_hole_area = self.params["min_hole_area"]
        if min_foreground_area < 1:
            min_foreground_area = int(min_foreground_area * mask.size)
        if min_hole_area < 1:
            min_hole_area = int(min_hole_area * mask.size)

        mode = cv2.RETR_CCOMP if detect_holes else cv2.RETR_EXTERNAL
        contours, hierarchy = cv2.findContours(
            mask.copy(), mode=mode, method=cv2.CHAIN_APPROX_NONE
        )

        if hierarchy is None:
            # no contours found --> return empty mask
            return []
        elif not detect_holes:
            ti = []
            tissue_id = 0
            for i, cnt in enumerate(contours):
                if cv2.contourArea(cnt) > min_foreground_area:
                    cnt = np.squeeze(cnt, axis=1)
                    # A polygon with less than 4 points is not valid
                    if len(cnt) >= 4:
                        ti.append(TissueInstance(id=tissue_id, contour=cnt, holes=[]))
                        tissue_id += 1
            return ti
        else:
            # separate outside and inside contours (region boundaries vs. holes in regions)
            # find the outside contours by looking for those with no parents (4th column is -1 if no parent)

            # TODO: Handle nested contours
            tissues = []
            for i, (cnt, hier) in enumerate(zip(contours, hierarchy[0])):
                # Check if the contour has a parent contour (i.e., if it's not a top-level contour)
                holes_ix = []
                if hier[3] == -1:
                    area = cv2.contourArea(cnt)
                    if area > min_foreground_area:
                        next_hole_index = hier[2]
                        # Iterate through the holes
                        while True:
                            # If it's a hole, add it to the list
                            if next_hole_index != -1:
                                next_hole = hierarchy[0][next_hole_index]
                                if (
                                    cv2.contourArea(contours[next_hole_index])
                                    > min_hole_area
                                ):
                                    holes_ix.append(next_hole_index)
                                next_hole_index = next_hole[0]
                            else:
                                break
                        tissues.append((i, holes_ix))

            ti = []
            for i, (tissue, holes) in enumerate(tissues):
                ti.append(
                    TissueInstance(
                        id=i,
                        contour=np.squeeze(contours[tissue], axis=1),
                        holes=[np.squeeze(contours[ix], axis=1) for ix in holes],
                    )
                )

            return ti
