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
    level : int, optional, default: None
        The level to use for segmentation.
    use_saturation : bool, optional, default: False
        Use saturation channel for segmentation.
    blur_ksize : int, optional, default: 17
    threshold : int, optional, default: 7
    morph_n_iter : int, optional, default: 3
    morph_k_size : int, optional, default: 7
    min_tissue_area : float, optional, default: 1e-3
        The minimum area of tissue.
    min_hole_area : float, optional, default: 1e-5
        The minimum area of holes.
    detect_holes : bool, optional, default: True
        Detect holes in tissue regions.
    filter_artifacts : bool, optional, default: True
        Filter artifacts out.
    key : str, optional, default: "tissue"
        The key to store the tissue contours.

    Examples
    --------

    .. plot::
        :context: close-figs

        >>> import lazyslide as zs
        >>> wsi = zs.WSI("https://github.com/camicroscope/Distro/raw/master/images/sample.svs")
        >>> zs.pp.find_tissue(wsi)
        >>> zs.pl.tissue(wsi)

    """
    # Get optimal level for segmentation
    if level is None:
        metadata = wsi.metadata

        warn = False
        if metadata.mpp is None:
            # Use the middle level
            level = metadata.n_level // 2
            warn = True
        else:
            search_space = np.asarray(metadata.level_downsample) * metadata.mpp
            level = np.argmin(np.abs(search_space - TARGET))

        # check if level is beyond the RAM
        current_shape = metadata.level_shape[level]
        # The data type in uint8, so each pixel is 1 byte
        # The size is calculated by width * height * 4 (RGBA)
        bytes_size = current_shape[0] * current_shape[1] * 4
        # if the size is beyond 4GB, use a higher level
        while bytes_size > 4e9:
            if level != metadata.n_level - 1:
                level += 1
                current_shape = metadata.level_shape[level]
                bytes_size = current_shape[0] * current_shape[1] * 4
            else:
                level = metadata.n_level - 1
                break
        if warn:
            warnings.warn(
                f"mpp is not available, " f"use level {level} for segmentation."
            )

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
        return False
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
        return False
    wsi.add_shapes(contours, data={"tissue_id": contours_ids}, name=f"{key}_contours")
    if len(holes) > 0:
        wsi.add_shapes(holes, data={"tissue_id": holes_ids}, name=f"{key}_holes")
