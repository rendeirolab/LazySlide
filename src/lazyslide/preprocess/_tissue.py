from __future__ import annotations

import logging
import warnings
from concurrent.futures import ProcessPoolExecutor
from typing import Sequence

import numpy as np
import pandas as pd
from shapely import Polygon

from lazyslide_cv.transform import TissueDetectionHE
from wsidata import WSIData

from lazyslide.preprocess._utils import get_scorer, Scorer
from lazyslide._utils import default_pbar
from lazyslide._const import Key

# TODO: Auto-selection of tissue level
#  should be decided by the RAM size
TARGET = 4  # mpp = 0.5 and downsample = 4


def find_tissue(
    wsi: WSIData,
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
    key: str = Key.tissue,
):
    """Find tissue regions in the WSI and add them as contours and holes.

    Parameters
    ----------
    wsi : WSIData
        Whole-slide image data object.
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
        >>> wsi = zs.open_wsi("https://github.com/camicroscope/Distro/raw/master/images/sample.svs")
        >>> zs.preprocess.find_tissue(wsi)
        >>> zs.plotting.tissue(wsi)

    """
    # Get optimal level for segmentation
    if level is None:
        metadata = wsi.properties

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
        downsample = wsi.properties.level_downsample[level]
    else:
        downsample = 1

    tissues, tissues_ids = [], []

    for tissue in tissue_instances:
        shell = tissue.contour * downsample
        holes = [hole * downsample for hole in tissue.holes]
        tissue_poly = Polygon(shell, holes=holes)
        tissues.append(tissue_poly)
        tissues_ids.append(tissue.id)

    if len(tissues) == 0:
        logging.warning("No tissue is found.")
        return False
    wsi.add_tissues(key=key, tissues=tissues, ids=tissues_ids)


def tissue_qc(
    wsi: WSIData,
    scores: Scorer | Sequence[Scorer],
    num_workers: int = 1,
    pbar: bool = True,
    key: str = Key.tissue,
    qc_key: str = Key.tissue_qc,
):
    compose_scorer = get_scorer(scores)

    with default_pbar(disable=not pbar) as progress_bar:
        task = progress_bar.add_task("Scoring tissue", total=wsi.n_tissue(key))
        scores = []
        qc = []

        if num_workers == 1:
            for tissue in wsi.iter.tissue_images(key, tissue_mask=True):
                result = compose_scorer(tissue.image, mask=tissue.mask)
                scores.append(result.scores)
                qc.append(result.qc)
                progress_bar.update(task, advance=1)
        else:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                # map is used to keep the order of the results
                jobs = []
                for tissue in wsi.iter.tissue_images(key, tissue_mask=True):
                    jobs.append(
                        executor.submit(compose_scorer, tissue.image, mask=tissue.mask)
                    )

                for job in jobs:
                    result = job.result()
                    scores.append(result.scores)
                    qc.append(result.qc)
                    progress_bar.update(task, advance=1)
        progress_bar.refresh()

    scores = pd.DataFrame(scores).assign(**{qc_key: qc})
    wsi.update_shapes_data(key=key, data=scores)
