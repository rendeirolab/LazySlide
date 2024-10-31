from collections import namedtuple

import cv2
import numpy as np
from shapely import Polygon

from .base import Transform
from .blur import MedianBlur
from .morph import MorphClose, MorphOpen
from .threshold import ArtifactFilterThreshold, BinaryThreshold

TissueInstance = namedtuple("TissueInstance", ["id", "contour", "holes"])


class ForegroundDetection(Transform):
    """
    Foreground detection for binary masks. Identifies regions that have a total area greater than
    specified threshold. Supports including holes within foreground regions, or excluding holes
    above a specified area threshold.

    Parameters
    ----------
    min_tissue_area : int or within (0, 1)
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
        min_tissue_area=0.01,
        min_hole_area=0.0001,
        detect_holes=True,
    ):
        self.set_params(
            min_tissue_area=min_tissue_area,
            min_hole_area=min_hole_area,
            detect_holes=detect_holes,
        )

    def __call__(self, mask):
        if isinstance(mask, np.ndarray):
            mask = mask.astype(np.uint8)
        return self.apply(mask)

    def apply(self, mask):
        detect_holes = self.params["detect_holes"]
        min_tissue_area = self.params["min_tissue_area"]
        min_hole_area = self.params["min_hole_area"]
        if min_tissue_area < 1:
            min_tissue_area = int(min_tissue_area * mask.size)
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
                if cv2.contourArea(cnt) > min_tissue_area:
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
                    if area > min_tissue_area:
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
            min_tissue_area=min_tissue_area,
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
                min_tissue_area=min_tissue_area,
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
