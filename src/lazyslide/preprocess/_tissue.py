from __future__ import annotations

import logging
import warnings
from concurrent.futures import ProcessPoolExecutor
from typing import Sequence

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import psutil
from shapely.affinity import scale, translate
from wsidata import WSIData
from wsidata.io import add_tissues, update_shapes_data

from lazyslide.cv.mask import BinaryMask
from lazyslide.cv.transform import (
    ArtifactFilterThreshold,
    BinaryThreshold,
    Compose,
    MedianBlur,
    MorphClose,
)

from .._const import Key
from .._utils import default_pbar, find_stack_level
from ..cv import merge_connected_polygons
from ._utils import Scorer, get_scorer


def _tissue_mask(
    image,
    to_hsv,
    filter_artifacts: bool = True,
    blur_ksize: int = 17,
    threshold: int = 7,
    morph_ksize: int = 7,
    morph_n_iter: int = 3,
):
    # Process image
    if not filter_artifacts:
        if to_hsv:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)[:, :, 1]
        else:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Decider the thresher
    if filter_artifacts:
        thresher = ArtifactFilterThreshold(threshold=threshold)
    else:
        if threshold is None:
            thresher = BinaryThreshold(use_otsu=True)
        else:
            thresher = BinaryThreshold(use_otsu=False, threshold=threshold)

    c = Compose(
        [
            MedianBlur(kernel_size=blur_ksize),
            thresher,
            # MorphOpen(kernel_size=morph_ksize, n_iterations=morph_n_iter),
            MorphClose(kernel_size=morph_ksize, n_iterations=morph_n_iter),
        ]
    )
    return c.apply(image)


def find_tissues(
    wsi: WSIData,
    level: int | str = "auto",
    refine_level: int | str | None = None,
    to_hsv: bool = False,
    blur_ksize: int = 17,
    threshold: int = 7,
    morph_n_iter: int = 3,
    morph_ksize: int = 7,
    min_tissue_area: float = 1e-3,
    min_hole_area: float = 1e-5,
    detect_holes: bool = True,
    filter_artifacts: bool = True,
    key_added: str = Key.tissue,
):
    """Find tissue regions in the WSI and add them as contours and holes.

    .. note::
        The results may not be deterministic between runs,
        as the segmentation level is automatically decided by the available memory.
        To get a consistent result, you can set the `level` parameter to a specific value.
        Set `level=-1` for the lowest resolution level and fastest segmentation speed.

    .. seealso::
        :func:`zs.seg.tissue <lazyslide.seg.tissue>`


    Parameters
    ----------
    wsi : :class:`WSIData <wsidata.WSIData>`
        The WSIData object to work on.
    level : int, default: 'auto'
        The level to use for segmentation.
    refine_level : int or 'auto', default: None
        The level to refine the tissue polygons.
    to_hsv : bool, default: False
        The tissue image will be converted from RGB to HSV space,
        the saturation channel (color purity) will be used for tissue detection.
    blur_ksize : int, default: 17
        The kernel size used to apply median blurring.
    threshold : int, default: 7
        The threshold for binary thresholding.
    morph_n_iter : int, default: 3
        The number of iterations of morphological opening and closing to apply.
    morph_ksize : int, default: 7
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
    :class:`GeoDataFrame <geopandas.GeoDataFrame>`
        The tissues dataframe, with columns of :code:`tissue_id` and :code:`geometry`.
        Added to :bdg-danger:`shapes`.

    Examples
    --------

    .. plot::
        :context: close-figs

        >>> import lazyslide as zs
        >>> wsi = zs.datasets.sample(with_data=False)
        >>> zs.pp.find_tissues(wsi)
        >>> zs.pl.tissue(wsi)

    """
    detect_holes_1 = detect_holes
    if refine_level is None:
        # If not refine, we can use a higher proportion of the memory
        proportion = 0.8
    else:
        # If we refine, we will do a quick search to the bounding box of the tissue regions
        proportion = 0.4
        detect_holes_1 = False

    ops_level = _decide_level(wsi, level, proportion)
    # Set the segmentation options

    # Run the first segmentation
    mask_option = dict(
        to_hsv=to_hsv,
        filter_artifacts=filter_artifacts,
        blur_ksize=blur_ksize,
        threshold=threshold,
        morph_ksize=morph_ksize,
        morph_n_iter=morph_n_iter,
    )
    to_poly_option = dict(
        min_area=min_tissue_area,
        min_hole_area=min_hole_area,
    )
    tissue_image = wsi.reader.get_level(ops_level)
    tissue_mask = _tissue_mask(tissue_image, **mask_option)
    tissue_polys = BinaryMask(tissue_mask).to_polygons(
        **to_poly_option, detect_holes=detect_holes_1
    )
    tissue_polys = tissue_polys.geometry

    if len(tissue_polys) == 0:
        logging.warning("No tissue is found.", stacklevel=find_stack_level())
        return False

    tissues = []
    downsample = _get_downsample(wsi, ops_level)
    for tissue in tissue_polys:
        # Scale it back to level 0
        tissue = scale(tissue, xfact=downsample, yfact=downsample, origin=(0, 0))
        tissues.append(tissue)

    if refine_level is not None:
        # Refine the tissue polygons at a higher resolution level
        refine_tissues = []
        for tissue_poly in tissues:
            # Tissue polygon at the highest resolution level
            xmin, ymin, xmax, ymax = tissue_poly.bounds
            # Enlarge the bounding box by 10%
            width, height = xmax - xmin, ymax - ymin

            if refine_level == "auto":
                current_refine_level = _decide_level(wsi, refine_level, proportion)
                if current_refine_level == ops_level:
                    current_refine_level -= 1
                if current_refine_level < 0:
                    current_refine_level = 0

            else:
                current_refine_level = refine_level

            refine_downsample = _get_downsample(wsi, current_refine_level)

            image = wsi.reader.get_region(
                xmin, ymin, width, height, level=current_refine_level
            )
            tissue_mask = _tissue_mask(image, **mask_option)
            tissue_polys = BinaryMask(tissue_mask).to_polygons(
                **to_poly_option, detect_holes=detect_holes
            )
            tissue_polys = tissue_polys.geometry

            for tissue in tissue_polys:
                tissue = scale(
                    tissue,
                    xfact=refine_downsample,
                    yfact=refine_downsample,
                    origin=(0, 0),
                )
                tissue = translate(tissue, xoff=xmin, yoff=ymin)
                refine_tissues.append(tissue.buffer(0))
        tissues_gdf = gpd.GeoDataFrame(
            data={"geometry": refine_tissues},
        )
        merged_tissue = merge_connected_polygons(tissues_gdf)
        tissues = merged_tissue["geometry"]

    add_tissues(wsi, key=key_added, tissues=tissues)


def score_tissues(
    wsi: WSIData,
    scorers: Scorer | Sequence[Scorer] = None,
    num_workers: int = 1,
    pbar: bool = False,
    tissue_key: str = Key.tissue,
):
    """Score tissue regions in the WSI for QC

    This is useful to filter out artifacts or non-tissue regions.

    .. deprecated:: 0.7.2

    Parameters
    ----------
    wsi : :class:`WSIData <wsidata.WSIData>`
        The WSIData object to work on.
    scorers : :class:`Scorer` or array of :class:`Scorer`
        :class:`Scorer` to use for scoring tissue regions.:

        - redness: The redness of the tissue.
        - brightness: The brightness of the tissue.
    num_workers : int, optional, default: 1
        Number of workers to use for scoring.
    pbar : bool, optional, default: False
        Show progress bar.
    tissue_key : str, optional, default: 'tissue'
        Key of the tissue data in the :bdg-danger:`shapes` slot.

    Returns
    -------
    None

    .. note::
        The scores will be added to the :code:`tissues | {tissue_key}` table in the WSIData object.
        The columns will be named after the scorers, e.g. `redness`, `brightness`.

    Examples
    --------

    .. code::

        >>> import lazyslide as zs
        >>> wsi = zs.datasets.sample(with_data=False)
        >>> zs.pp.find_tissues(wsi)
        >>> zs.pp.score_tiles(wsi, ["redness", "brightness"])
        >>> wsi["tissues"]


    """

    warnings.warn(
        "This function is deprecated and will be removed after v0.9.0",
        stacklevel=find_stack_level(),
    )

    if scorers is None:
        scorers = ["redness", "brightness"]
    compose_scorer = get_scorer(scorers)

    with default_pbar(disable=not pbar) as progress_bar:
        task = progress_bar.add_task(
            "Scoring tissue", total=wsi.fetch.n_tissue(tissue_key)
        )
        scores = []
        qc = []

        if num_workers == 1:
            for tissue in wsi.iter.tissue_images(tissue_key, mask_bg=True):
                result = compose_scorer(tissue.image, mask=tissue.mask)
                scores.append(result.scores)
                qc.append(result.qc)
                progress_bar.update(task, advance=1)
        else:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                # map is used to keep the order of the results
                jobs = []
                for tissue in wsi.iter.tissue_images(tissue_key, mask_bg=True):
                    jobs.append(
                        executor.submit(compose_scorer, tissue.image, mask=tissue.mask)
                    )

                for job in jobs:
                    result = job.result()
                    scores.append(result.scores)
                    qc.append(result.qc)
                    progress_bar.update(task, advance=1)
        progress_bar.refresh()

    scores = pd.DataFrame(scores)  # .assign(**{qc_key: qc})
    update_shapes_data(wsi, key=tissue_key, data=scores)


def _get_optimal_level(metadata, in_bounds=True, proportion=0.8):
    # Get optimal level for segmentation
    # Current available memory
    available_memory = psutil.virtual_memory().available * proportion  # in bytes

    warn = False
    if metadata.mpp is None:
        # Use the middle level
        level = metadata.n_level // 2
        warn = True
    else:
        search_space = np.asarray(metadata.level_downsample) * metadata.mpp
        level = np.argmin(np.abs(search_space - 4))

    # check if level is beyond the RAM
    current_shape = metadata.level_shape[level]
    # The data type in uint8, so each pixel is 1 byte
    # The size is calculated by width * height * 4 (RGBA)
    bytes_size = current_shape[0] * current_shape[1] * 4
    # if the size is beyond 4GB, use a higher level
    while bytes_size > available_memory:
        if level != metadata.n_level - 1:
            level += 1
            current_shape = metadata.level_shape[level]
            bytes_size = current_shape[0] * current_shape[1] * 4
        else:
            level = metadata.n_level - 1
            break
    if warn:
        warnings.warn(f"mpp is not available, use level {level} for segmentation.")
    return level


def _decide_level(wsi, level, proportion=0.8):
    if level == "auto":
        return _get_optimal_level(wsi.properties, proportion)
    else:
        return wsi.reader.translate_level(level)


def _get_downsample(wsi, level):
    if level == 0:
        return 1
    else:
        return wsi.properties.level_downsample[level]
