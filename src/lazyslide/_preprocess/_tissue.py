from __future__ import annotations

import logging
import warnings
from concurrent.futures import ProcessPoolExecutor
from typing import Sequence

import numpy as np
import pandas as pd
from shapely import Polygon
from wsidata import WSIData
from wsidata.io import add_tissues, update_shapes_data

from ._utils import get_scorer, Scorer
from .._const import Key
from .._cv.transform import TissueDetectionHE
from .._utils import default_pbar

# TODO: Auto-selection of tissue level
#  should be decided by the RAM size
TARGET = 4  # mpp = 0.5 and downsample = 4

# TODO: TO have a better hole detection
# We can first identify the tissue regions at the least resolution level
# Then we get the bbox of the tissue regions, and then we can go to higher resolution levels


def find_tissues(
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
    key_added: str = Key.tissue,
):
    """Find tissue regions in the WSI and add them as contours and holes.

    Parameters
    ----------
    wsi : :class:`WSIData <wsidata.WSIData>`
        The WSIData object to work on.
    level : int, default: None
        The level to use for segmentation.
    use_saturation : bool, default: False
        The tissue image will be converted from RGB to HSV space,
        the saturation channel (color purity) will be used for tissue detection.
    blur_ksize : int, default: 17
        The kernel size used to apply median blurring.
    threshold : int, default: 7
        The threshold for binary thresholding.
    morph_n_iter : int, default: 3
        The number of iterations of morphological opening and closing to apply.
    morph_k_size : int, default: 7
        The kernel size for morphological opening and closing.
    min_tissue_area : float, default: 1e-3
        The minimum area of tissue.
    min_hole_area : float, default: 1e-5
        The minimum area of holes.
    detect_holes : bool, default: True
        Detect holes in tissue regions.
    filter_artifacts : bool, default: True
        Filter artifacts out. Artifacts that are non-redish are removed.
    key_added : str, default: 'tissues'
        The key to save the result in the WSIData object.

    Returns
    -------
    - A :class:`GeoDataFrame <geopandas.GeoDataFrame>` with columns of 'tissue_id' and 'geometry'.
      added to the :bdg-danger:`shapes` slot of the SpatialData object.


    Examples
    --------

    .. plot::
        :context: close-figs

        >>> from wsidata import open_wsi
        >>> import lazyslide as zs
        >>> wsi = open_wsi("https://github.com/camicroscope/Distro/raw/master/images/sample.svs")
        >>> zs.pp.find_tissues(wsi)
        >>> zs.pl.tissue(wsi)

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
        if len(tissue.holes) == 0:
            tissue_poly = Polygon(shell)
        else:
            holes = [hole * downsample for hole in tissue.holes]
            tissue_poly = Polygon(shell, holes=holes)
        tissues.append(tissue_poly)
        tissues_ids.append(tissue.id)

    if len(tissues) == 0:
        logging.warning("No tissue is found.")
        return False
    add_tissues(wsi, key=key_added, tissues=tissues, ids=tissues_ids)


def tissues_qc(
    wsi: WSIData,
    scores: Scorer | Sequence[Scorer] = None,
    num_workers: int = 1,
    pbar: bool = False,
    tissue_key: str = Key.tissue,
    qc_key: str = Key.tissue_qc,
):
    """Score tissue regions in the WSI for QC

    This is useful to filter out artifacts or non-tissue regions.

    Parameters
    ----------
    wsi : :class:`WSIData <wsidata.WSIData>`
        The WSIData object to work on.
    scores : :class:`Scorer` or array of :class`Scorer`
        :class:`Scorer` to use for scoring tissue regions.
        - 'redness': The redness of the tissue.
        - 'brightness': The brightness of the tissue.
    num_workers : int, optional, default: 1
        Number of workers to use for scoring.
    pbar : bool, optional, default: False
        Show progress bar.
    tissue_key : str, optional, default: 'tissue'
        Key of the tissue data in the :bdg-danger:`shapes` slot.
    qc_key : str, optional, default: 'qc'
        The key in the tissue dataframe indicates if a tissue passed qc.

    Examples
    --------

    .. code::

        >>> from wsidata import open_wsi
        >>> import lazyslide as zs
        >>> wsi = open_wsi("https://github.com/camicroscope/Distro/raw/master/images/sample.svs")
        >>> zs.pp.find_tissues(wsi)
        >>> zs.pp.tissues_qc(wsi, ["redness", "brightness"])
        >>> wsi["tissues"]


    """
    if scores is None:
        scores = ["redness", "brightness"]
    compose_scorer = get_scorer(scores)

    with default_pbar(disable=not pbar) as progress_bar:
        task = progress_bar.add_task(
            "Scoring tissue", total=wsi.fetch.n_tissue(tissue_key)
        )
        scores = []
        qc = []

        if num_workers == 1:
            for tissue in wsi.iter.tissue_images(tissue_key, tissue_mask=True):
                result = compose_scorer(tissue.image, mask=tissue.mask)
                scores.append(result.scores)
                qc.append(result.qc)
                progress_bar.update(task, advance=1)
        else:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                # map is used to keep the order of the results
                jobs = []
                for tissue in wsi.iter.tissue_images(tissue_key, tissue_mask=True):
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
    update_shapes_data(wsi, key=tissue_key, data=scores)
