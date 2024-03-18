# See https://github.com/Dana-Farber-AIOS/pathml/blob/master/pathml/preprocessing/transforms.py
import cv2
import numpy as np

from .base import Transform


class ConvertColorspace(Transform):
    def __init__(
        self,
        code=None,
        old=None,
        new=None,
    ):
        if code is None:
            self.old = old.upper()
            self.new = new.upper()
            if code is None:
                code = getattr(cv2, f"COLOR_{self.old}2{self.new}".upper())
        self.code = code

    def apply(self, image):
        return cv2.cvtColor(image, self.code)


class MedianBlur(Transform):
    """
    Median blur kernel.

    Args:
        kernel_size (int): Width of kernel. Must be an odd number. Defaults to 5.
    """

    def __init__(self, kernel_size=5):
        assert kernel_size % 2 == 1, "kernel_size must be an odd number"
        self.kernel_size = kernel_size

    def __repr__(self):
        return f"MedianBlur(kernel_size={self.kernel_size})"

    def apply(self, image):
        assert image.dtype == np.uint8, f"image dtype {image.dtype} must be np.uint8"
        return cv2.medianBlur(image, ksize=self.kernel_size)


class GaussianBlur(Transform):
    """
    Gaussian blur kernel.

    Args:
        kernel_size (int): Width of kernel. Must be an odd number. Defaults to 5.
        sigma (float): Variance of Gaussian kernel. Variance is assumed to be equal in X and Y axes. Defaults to 5.
    """

    def __init__(self, kernel_size=5, sigma=5):
        self.k_size = kernel_size
        self.sigma = sigma

    def __repr__(self):
        return f"GaussianBlur(kernel_size={self.k_size}, sigma={self.sigma})"

    def apply(self, image):
        assert image.dtype == np.uint8, f"image dtype {image.dtype} must be np.uint8"
        out = cv2.GaussianBlur(
            image,
            ksize=(self.k_size, self.k_size),
            sigmaX=self.sigma,
            sigmaY=self.sigma,
        )
        return out


class BoxBlur(Transform):
    """
    Box (average) blur kernel.

    Args:
        kernel_size (int): Width of kernel. Defaults to 5.
    """

    def __init__(self, kernel_size=5):
        self.kernel_size = kernel_size

    def __repr__(self):
        return f"BoxBlur(kernel_size={self.kernel_size})"

    def F(self, image):
        assert image.dtype == np.uint8, f"image dtype {image.dtype} must be np.uint8"
        return cv2.boxFilter(
            image, ksize=(self.kernel_size, self.kernel_size), ddepth=-1
        )


class BinaryThreshold(Transform):
    """
    Binary thresholding transform to create a binary mask.
    If input image is RGB it is first converted to greyscale, otherwise the input must have 1 channel.

    Args:
        mask_name (str): Name of mask that is created.
        use_otsu (bool): Whether to use Otsu's method to automatically determine optimal threshold. Defaults to True.
        threshold (int): Specified threshold. Ignored if ``use_otsu is True``. Defaults to 0.
        inverse (bool): Whether to use inverse threshold. If using inverse threshold, pixels below the threshold will
            be returned as 1. Otherwise, pixels below the threshold will be returned as 0. Defaults to ``False``.

    References:
        Otsu, N., 1979. A threshold selection method from gray-level histograms. IEEE transactions on systems,
        man, and cybernetics, 9(1), pp.62-66.
    """

    def __init__(self, mask_name=None, use_otsu=True, threshold=0, inverse=False):
        self.threshold = threshold
        self.max_value = 255
        self.use_otsu = use_otsu
        self.inverse = inverse
        self.mask_name = mask_name
        self.type = cv2.THRESH_BINARY_INV if inverse else cv2.THRESH_BINARY
        if use_otsu:
            self.type += cv2.THRESH_OTSU

    def __repr__(self):
        return (
            f"BinaryThreshold(use_otsu={self.use_otsu}, threshold={self.threshold}, "
            f"mask_name={self.mask_name}, inverse={self.inverse})"
        )

    def apply(self, image):
        assert image.dtype == np.uint8, f"image dtype {image.dtype} must be np.uint8"
        assert (
            image.ndim == 2
        ), f"input image has shape {image.shape}. Must convert to 1-channel image (H, W)."
        _, out = cv2.threshold(
            src=image,
            thresh=self.threshold,
            maxval=self.max_value,
            type=self.type,
        )
        return out.astype(np.uint8)


class MorphOpen(Transform):
    """
    Morphological opening. First applies erosion operation, then dilation.
    Reduces noise by removing small objects from the background.
    Operates on a binary mask.

    Args:
        mask_name (str): Name of mask on which to apply transform
        kernel_size (int): Size of kernel for default square kernel. Ignored if a custom kernel is specified.
            Defaults to 5.
        n_iterations (int): Number of opening operations to perform. Defaults to 1.
    """

    def __init__(self, mask_name=None, kernel_size=5, n_iterations=1):
        self.kernel_size = kernel_size
        self.n_iterations = n_iterations
        self.mask_name = mask_name

    def __repr__(self):
        return (
            f"MorphOpen(kernel_size={self.kernel_size}, n_iterations={self.n_iterations}, "
            f"mask_name={self.mask_name})"
        )

    def apply(self, mask):
        assert mask.dtype == np.uint8, f"mask type {mask.dtype} must be np.uint8"
        k = np.ones((self.kernel_size, self.kernel_size), dtype=np.uint8)
        out = cv2.morphologyEx(
            src=mask, kernel=k, op=cv2.MORPH_OPEN, iterations=self.n_iterations
        )
        return out


class MorphClose(Transform):
    """
    Morphological closing. First applies dilation operation, then erosion.
    Reduces noise by closing small holes in the foreground.
    Operates on a binary mask.

    Args:
        mask_name (str): Name of mask on which to apply transform
        kernel_size (int): Size of kernel for default square kernel. Ignored if a custom kernel is specified.
            Defaults to 5.
        n_iterations (int): Number of opening operations to perform. Defaults to 1.
    """

    def __init__(self, mask_name=None, kernel_size=5, n_iterations=1):
        self.kernel_size = kernel_size
        self.n_iterations = n_iterations
        self.mask_name = mask_name

    def __repr__(self):
        return (
            f"MorphClose(kernel_size={self.kernel_size}, n_iterations={self.n_iterations}, "
            f"mask_name={self.mask_name})"
        )

    def apply(self, mask):
        assert mask.dtype == np.uint8, f"mask type {mask.dtype} must be np.uint8"
        k = np.ones((self.kernel_size, self.kernel_size), dtype=np.uint8)
        out = cv2.morphologyEx(
            src=mask, kernel=k, op=cv2.MORPH_CLOSE, iterations=self.n_iterations
        )
        return out


class ForegroundDetection(Transform):
    """
    Foreground detection for binary masks. Identifies regions that have a total area greater than
    specified threshold. Supports including holes within foreground regions, or excluding holes
    above a specified area threshold.

    Args:
        min_region_size (int): Minimum area of detected foreground regions, in pixels. Defaults to 5000.
        max_hole_size (int): Maximum size of allowed holes in foreground regions, in pixels.
            Ignored if ``outer_contours_only is True``. Defaults to 1500.
        outer_contours_only (bool): If true, ignore holes in detected foreground regions. Defaults to False.
        mask_name (str): Name of mask on which to apply transform

    References:
        Lu, M.Y., Williamson, D.F., Chen, T.Y., Chen, R.J., Barbieri, M. and Mahmood, F., 2020. Data Efficient and
        Weakly Supervised Computational Pathology on Whole Slide Images. arXiv preprint arXiv:2004.09666.
    """

    def __init__(
        self,
        mask_name=None,
        min_region_size=5000,
        max_hole_size=1500,
        outer_contours_only=False,
    ):
        self.min_region_size = min_region_size
        self.max_hole_size = max_hole_size
        self.outer_contours_only = outer_contours_only
        self.mask_name = mask_name

    def __repr__(self):
        return (
            f"ForegroundDetection(min_region_size={self.min_region_size}, max_hole_size={self.max_hole_size},"
            f"outer_contours_only={self.outer_contours_only}, mask_name={self.mask_name})"
        )

    def apply(self, mask):
        assert mask.dtype == np.uint8, f"mask type {mask.dtype} must be np.uint8"
        mode = cv2.RETR_EXTERNAL if self.outer_contours_only else cv2.RETR_CCOMP
        contours, hierarchy = cv2.findContours(
            mask.copy(), mode=mode, method=cv2.CHAIN_APPROX_NONE
        )

        if hierarchy is None:
            # no contours found --> return empty mask
            mask_out = np.zeros_like(mask)
        elif self.outer_contours_only:
            out = np.zeros_like(mask, dtype=np.int8)
            for c in contours:
                # ignore contours below size threshold
                if cv2.contourArea(c) > self.min_region_size:
                    # fill contours
                    cv2.fillPoly(out, [c], 255)
            mask_out = out
        else:
            # separate outside and inside contours (region boundaries vs. holes in regions)
            # find the outside contours by looking for those with no parents (4th column is -1 if no parent)
            hierarchy = np.squeeze(hierarchy, axis=0)
            outside_contours = hierarchy[:, 3] == -1
            hole_contours = ~outside_contours

            # outside contours must be above min_tissue_region_size threshold
            contour_size_thresh = [
                cv2.contourArea(c) > self.min_region_size for c in contours
            ]
            # hole contours must be above area threshold
            hole_size_thresh = [
                cv2.contourArea(c) > self.max_hole_size for c in contours
            ]
            # holes must have parents above area threshold
            hole_parent_thresh = [
                p in np.argwhere(contour_size_thresh).flatten() for p in hierarchy[:, 3]
            ]

            outside_contours = np.array(outside_contours)
            hole_contours = np.array(hole_contours)
            contour_size_thresh = np.array(contour_size_thresh)
            hole_size_thresh = np.array(hole_size_thresh)
            hole_parent_thresh = np.array(hole_parent_thresh)

            # now combine outside and inside contours into final mask
            out1 = np.zeros_like(mask, dtype=np.int8)
            out2 = np.zeros_like(mask, dtype=np.int8)

            # loop thru contours
            for (
                cnt,
                outside,
                size_thresh,
                hole,
                hole_size_thresh,
                hole_parent_thresh,
            ) in zip(
                contours,
                outside_contours,
                contour_size_thresh,
                hole_contours,
                hole_size_thresh,
                hole_parent_thresh,
            ):
                if outside and size_thresh:
                    # in this case, the contour is an outside contour
                    cv2.fillPoly(out1, [cnt], 255)
                if hole and hole_size_thresh and hole_parent_thresh:
                    # in this case, the contour is an inside contour
                    cv2.fillPoly(out2, [cnt], 255)

            mask_out = out1 - out2

        return mask_out.astype(np.uint8)


class ForegroundContourDetection(Transform):
    """
    Foreground detection for binary masks. Identifies regions that have a total area greater than
    specified threshold. Supports including holes within foreground regions, or excluding holes
    above a specified area threshold.

    Args:
        min_region_size (int): Minimum area of detected foreground regions, in pixels. Defaults to 5000.
        max_hole_size (int): Maximum size of allowed holes in foreground regions, in pixels.
            Ignored if ``outer_contours_only is True``. Defaults to 1500.
        outer_contours_only (bool): If true, ignore holes in detected foreground regions. Defaults to False.
        mask_name (str): Name of mask on which to apply transform

    References:
        Lu, M.Y., Williamson, D.F., Chen, T.Y., Chen, R.J., Barbieri, M. and Mahmood, F., 2020. Data Efficient and
        Weakly Supervised Computational Pathology on Whole Slide Images. arXiv preprint arXiv:2004.09666.
    """

    def __init__(
        self,
        mask_name=None,
        min_region_size=5000,
        max_hole_size=100,
        outer_contours_only=False,
    ):
        self.min_region_size = min_region_size
        self.max_hole_size = max_hole_size
        self.outer_contours_only = outer_contours_only
        self.mask_name = mask_name

    def __repr__(self):
        return (
            f"ForegroundDetection(min_region_size={self.min_region_size}, max_hole_size={self.max_hole_size},"
            f"outer_contours_only={self.outer_contours_only}, mask_name={self.mask_name})"
        )

    def apply(self, mask):
        assert mask.dtype == np.uint8, f"mask type {mask.dtype} must be np.uint8"
        mode = cv2.RETR_EXTERNAL if self.outer_contours_only else cv2.RETR_CCOMP
        contours, hierarchy = cv2.findContours(
            mask.copy(), mode=mode, method=cv2.CHAIN_APPROX_NONE
        )

        if hierarchy is None:
            # no contours found --> return empty mask
            return [], []
        elif self.outer_contours_only:
            return contours, []
        else:
            # separate outside and inside contours (region boundaries vs. holes in regions)
            # find the outside contours by looking for those with no parents (4th column is -1 if no parent)
            contours_ix = np.arange(len(contours))
            hierarchy = np.squeeze(hierarchy, axis=0)
            outmost_slice = hierarchy[:, 3] == -1
            hole_slice = ~outmost_slice

            outmost = contours_ix[outmost_slice]
            holes = contours_ix[hole_slice]
            contours_areas = np.array([cv2.contourArea(c) for c in contours])

            # outside contours must be above min_tissue_region_size threshold
            tissue_contours = outmost[
                contours_areas[outmost_slice] > self.min_region_size
            ]

            tissue_holes = holes[
                # hole contours must be above area threshold
                contours_areas[hole_slice]
                > self.max_hole_size
                &
                # holes must have parents above area threshold
                (contours_areas[hierarchy[hole_slice, 3]] > self.min_region_size)
            ]

            return (
                [np.squeeze(contours[ix], axis=1) for ix in tissue_contours],
                [np.squeeze(contours[ix], axis=1) for ix in tissue_holes],
            )


class TissueDetectionHE(Transform):
    """
    Detect tissue regions from H&E stained slide.
    First applies a median blur, then binary thresholding, then morphological opening and closing, and finally
    foreground detection.

    Args:
        use_saturation (bool): Whether to convert to HSV and use saturation channel for tissue detection.
            If False, convert from RGB to greyscale and use greyscale image_ref for tissue detection. Defaults to True.
        blur_ksize (int): kernel size used to apply median blurring. Defaults to 15.
        threshold (int): threshold for binary thresholding. If None, uses Otsu's method. Defaults to None.
        morph_n_iter (int): number of iterations of morphological opening and closing to apply. Defaults to 3.
        morph_k_size (int): kernel size for morphological opening and closing. Defaults to 7.
        min_region_size (int): Minimum area of detected foreground regions, in pixels. Defaults to 5000.
        max_hole_size (int): Maximum size of allowed holes in foreground regions, in pixels.
            Ignored if outer_contours_only=True. Defaults to 1500.
        outer_contours_only (bool): If true, ignore holes in detected foreground regions. Defaults to False.
    """

    def __init__(
        self,
        use_saturation=True,
        blur_ksize=17,
        threshold=7,
        morph_n_iter=3,
        morph_k_size=7,
        min_region_size=2500,
        max_hole_size=100,
        outer_contours_only=False,
        return_contours=False,
    ):
        self.use_sat = use_saturation
        self.blur_ksize = blur_ksize
        self.threshold = threshold
        self.morph_n_iter = morph_n_iter
        self.morph_k_size = morph_k_size
        self.min_region_size = min_region_size
        self.max_hole_size = max_hole_size
        self.outer_contours_only = outer_contours_only

        if self.threshold is None:
            thresholder = BinaryThreshold(use_otsu=True)
        else:
            thresholder = BinaryThreshold(use_otsu=False, threshold=self.threshold)

        if not return_contours:
            foreground = ForegroundDetection(
                min_region_size=self.min_region_size,
                max_hole_size=self.max_hole_size,
                outer_contours_only=self.outer_contours_only,
            )
        else:
            foreground = ForegroundContourDetection(
                min_region_size=self.min_region_size,
                max_hole_size=self.max_hole_size,
                outer_contours_only=self.outer_contours_only,
            )

        self.pipeline = [
            MedianBlur(kernel_size=self.blur_ksize),
            thresholder,
            MorphOpen(kernel_size=self.morph_k_size, n_iterations=self.morph_n_iter),
            MorphClose(kernel_size=self.morph_k_size, n_iterations=self.morph_n_iter),
            foreground,
        ]

    def __repr__(self):
        return (
            f"TissueDetectionHE(use_sat={self.use_sat}, blur_ksize={self.blur_ksize}, "
            f"threshold={self.threshold}, morph_n_iter={self.morph_n_iter}, "
            f"morph_k_size={self.morph_k_size}, min_region_size={self.min_region_size}, "
            f"max_hole_size={self.max_hole_size}, outer_contours_only={self.outer_contours_only})"
        )

    def apply(self, image):
        assert (
            image.dtype == np.uint8
        ), f"Input image dtype {image.dtype} must be np.uint8"
        # first get single channel image_ref
        if self.use_sat:
            one_channel = ConvertColorspace(code=cv2.COLOR_RGB2HSV).apply(image)
            one_channel = one_channel[:, :, 1]
        else:
            one_channel = ConvertColorspace(code=cv2.COLOR_RGB2GRAY).apply(image)

        tissue = one_channel
        for p in self.pipeline:
            tissue = p.apply(tissue)
        return tissue
