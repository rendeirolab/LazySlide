import cv2

from .mods import (
    ArtifactFilterThreshold,
    BinaryThreshold,
    ForegroundDetection,
    MedianBlur,
    MorphClose,
    Transform,
)


class TissueDetectionHE(Transform):
    """
    Detect tissue regions from H&E stained slide.
    First applies a median blur, then binary thresholding, then morphological opening and closing, and finally
    foreground detection.

    Parameters
    ----------
    use_saturation : bool
        Whether to convert to HSV and use saturation channel for tissue detection.
        If False, convert from RGB to greyscale and use greyscale image_ref for tissue detection. Defaults to True.
    blur_ksize : int
        kernel size used to apply median blurring. Defaults to 7.

        .. note::
            This default changed from 17 to 7, so that a default
            ``TissueDetectionHE`` keeps blurring with the same kernel it used
            when ``Transform.params`` was shared and the blur silently picked
            up ``morph_k_size``.
    threshold : int
        threshold for binary thresholding. If None, uses Otsu's method. Defaults to None.
    morph_n_iter : int
        number of iterations of morphological opening and closing to apply. Defaults to 3.
    morph_k_size : int
        kernel size for morphological opening and closing. Defaults to 7.
    min_region_size : int
    """

    def __init__(
        self,
        use_saturation=False,
        blur_ksize=7,
        threshold=7,
        morph_n_iter=3,
        morph_k_size=7,
        min_tissue_area=0.01,
        min_hole_area=0.0001,
        detect_holes=True,
        filter_artifacts=True,
    ):
        self.set_params(
            use_saturation=use_saturation,
            blur_ksize=blur_ksize,
            threshold=threshold,
            morph_n_iter=morph_n_iter,
            morph_k_size=morph_k_size,
            min_tissue_area=min_tissue_area,
            min_hole_area=min_hole_area,
            detect_holes=detect_holes,
            filter_artifacts=filter_artifacts,
        )

        if filter_artifacts:
            thresholder = ArtifactFilterThreshold(threshold=threshold)
        else:
            if threshold is None:
                thresholder = BinaryThreshold(use_otsu=True)
            else:
                thresholder = BinaryThreshold(use_otsu=False, threshold=threshold)

        foreground = ForegroundDetection(
            min_foreground_area=min_tissue_area,
            min_hole_area=min_hole_area,
            detect_holes=detect_holes,
        )

        self.pipeline = [
            MedianBlur(kernel_size=blur_ksize),
            thresholder,
            MorphClose(kernel_size=morph_k_size, n_iterations=morph_n_iter),
            foreground,
        ]

    def apply(self, image):
        filter_artifacts = self.params["filter_artifacts"]
        use_saturation = self.params["use_saturation"]

        if not filter_artifacts:
            if use_saturation:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)[:, :, 1]
            else:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        for p in self.pipeline:
            image = p.apply(image)
        return image
