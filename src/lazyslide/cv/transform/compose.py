import cv2
from shapely import Polygon

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
        kernel size used to apply median blurring. Defaults to 15.
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
        blur_ksize=17,
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
            # MorphOpen(kernel_size=morph_k_size, n_iterations=morph_n_iter),
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


class Mask2Polygon(Transform):
    """
    Convert binary mask to polygon.

    Parameters
    ----------
    min_area : int
        Minimum area of detected regions to be included in the polygon.
    """

    def __init__(
        self,
        min_area=0,
        morph_k_size=7,
        morph_n_iter=3,
        min_tissue_area=0.01,
        min_hole_area=0.0001,
        detect_holes=True,
    ):
        self.set_params(min_area=min_area)

        self.pipeline = [
            # MorphOpen(kernel_size=morph_k_size, n_iterations=morph_n_iter),
            MorphClose(kernel_size=morph_k_size, n_iterations=morph_n_iter),
            ForegroundDetection(
                min_foreground_area=min_tissue_area,
                min_hole_area=min_hole_area,
                detect_holes=detect_holes,
            ),
        ]

    def apply(self, mask):
        min_area = self.params["min_area"]

        for p in self.pipeline:
            try:
                mask = p.apply(mask)
            except Exception as e:
                print(self.__class__.__name__, e)

        tissue_instances = mask
        polygons = []
        if len(tissue_instances) == 0:
            return []
        for tissue in tissue_instances:
            shell = tissue.contour
            if len(tissue.holes) == 0:
                tissue_poly = Polygon(shell)
            else:
                holes = [hole for hole in tissue.holes]
                tissue_poly = Polygon(shell, holes=holes)
            if tissue_poly.area < min_area:
                continue
            polygons.append(tissue_poly)
        return polygons
