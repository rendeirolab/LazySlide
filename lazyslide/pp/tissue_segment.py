from __future__ import annotations

import logging
import warnings

import numpy as np

from lazyslide.cv.transform import TissueDetectionHE
from lazyslide.wsi import WSI

# TODO: Auto-selection of tissue level
#  should be decided by the RAM size
TARGET = 4  # mpp = 0.5 and downsample = 4


def find_tissue(
    wsi: WSI,
    level: int | None = None,
    use_saturation: bool = False,
    blur_ksize: int = 17,
    threshold: int = 7,
    morph_n_iter: int = 3,
    morph_k_size: int = 7,
    min_tissue_area: float = 1e-3,
    min_hole_area: float = 1e-5,
    detect_holes: bool = True,
    filter_artifacts: bool = True,
    key: str = "tissue",
):
    """Find tissue regions in the WSI and add them as contours and holes.

    Parameters
    ----------
    wsi : WSI
        Whole-slide image object.
    level : int, optional
        The level to use for segmentation, by default None.

    """
    # Get optimal level for segmentation
    if level is None:
        metadata = wsi.metadata
        if metadata.mpp is None:
            # Use the middle level
            level = metadata.n_level // 2
            warnings.warn(
                f"mpp is not available, " f"use level {level} for segmentation."
            )
        else:
            search_space = np.asarray(metadata.level_downsample) * metadata.mpp
            level = np.argmin(np.abs(search_space - TARGET))
    else:
        level = wsi.reader.translate_level(level)

    seg = TissueDetectionHE(
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
    image = wsi.reader.get_level(level)
    tissue_instances = seg.apply(image)
    if len(tissue_instances) == 0:
        logging.warning("No tissue is found.")
        return
    if level != 0:
        downsample = wsi.metadata.level_downsample[level]
    else:
        downsample = 1

    contours, holes = [], []
    contours_ids, holes_ids = [], []

    for tissue in tissue_instances:
        contours.append((tissue.contour * downsample))
        contours_ids.append(tissue.id)
        for hole in tissue.holes:
            holes.append((hole * downsample))
            holes_ids.append(tissue.id)

    if len(contours) == 0:
        logging.warning("No tissue is found.")
        return
    wsi.add_contours(contours, data={"tissue_id": contours_ids}, name=f"{key}_contours")
    if len(holes) > 0:
        wsi.add_holes(holes, data={"tissue_id": holes_ids}, name=f"{key}_holes")
