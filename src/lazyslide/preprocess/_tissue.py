from __future__ import annotations

import logging
import warnings
from concurrent.futures import ProcessPoolExecutor
from typing import Sequence

import cv2
import numpy as np
import pandas as pd
import psutil
from shapely import Polygon
from shapely.affinity import translate, scale
from wsidata import WSIData
from wsidata.io import add_tissues, update_shapes_data

from lazyslide.cv.mask import BinaryMask
from lazyslide.cv.transform import (
    ArtifactFilterThreshold,
    BinaryThreshold,
    MedianBlur,
    MorphClose,
    Compose,
)
from ._utils import get_scorer, Scorer
from .._const import Key
from .._utils import default_pbar, find_stack_level


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

    Parameters
    ----------
    wsi : :class:`WSIData <wsidata.WSIData>`
        The WSIData object to work on.
    level : int, default: 'auto'
        The level to use for segmentation.
    refine_level : int or 'auto', default: None
        The level to refine the tissue polygons.
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
        >>> wsi = open_wsi("sample.svs")
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

    level = _decide_level(wsi, level, proportion)
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
    tissue_image = wsi.reader.get_level(level)
    tissue_mask = _tissue_mask(tissue_image, **mask_option)
    tissue_polys = BinaryMask(tissue_mask).to_polygons(
        **to_poly_option, detect_holes=detect_holes_1
    )

    if len(tissue_polys) == 0:
        logging.warning("No tissue is found.", stacklevel=find_stack_level())
        return False

    tissues = []
    downsample = _get_downsample(wsi, level)
    for tissue in tissue_polys:
        # Scale it back to level 0
        tissue = scale(tissue, xfact=downsample, yfact=downsample, origin=(0, 0))
        tissues.append(tissue)

    if refine_level:
        # Refine the tissue polygons at a higher resolution level
        refine_tissues = []
        for tissue_poly in tissues:
            # Tissue polygon at the highest resolution level
            xmin, ymin, xmax, ymax = tissue_poly.buffer(10).bounds
            width, height = xmax - xmin, ymax - ymin

            level = _decide_level(wsi, refine_level)
            refine_downsample = _get_downsample(wsi, level)

            image = wsi.reader.get_region(xmin, ymin, width, height, level=level)
            tissue_mask = _tissue_mask(image, **mask_option)
            tissue_polys = BinaryMask(tissue_mask).to_polygons(
                **to_poly_option, detect_holes=detect_holes
            )

            for tissue in tissue_polys:
                tissue = scale(
                    tissue,
                    xfact=refine_downsample,
                    yfact=refine_downsample,
                    origin=(0, 0),
                )
                tissue = translate(tissue, xoff=xmin, yoff=ymin)
                refine_tissues.append(tissue)
            # refine_tissue_instances = seg.apply(image)
            # for refine_tissue in refine_tissue_instances:
            #     refine_poly = _tissue_instance2poly(
            #         refine_tissue, refine_downsample, xoff=xmin, yoff=ymin
            #     )
            #
            #     if not refine_poly.is_valid:
            #         refine_poly = refine_poly.buffer(0)
            #     if isinstance(refine_poly, MultiPolygon):
            #         geoms = refine_poly.geoms
            #         areas = [geom.area for geom in geoms]
            #         refine_poly = geoms[np.argmax(areas)]
            #
            #     if refine_poly.is_valid:
            #         # Check if the refined tissue region intersects with the original tissue region
            #         if refine_poly.intersects(tissue_poly):
            #             tissues.append(refine_poly)
        tissues = refine_tissues

    add_tissues(wsi, key=key_added, tissues=tissues)


def score_tissues(
    wsi: WSIData,
    scores: Scorer | Sequence[Scorer] = None,
    num_workers: int = 1,
    pbar: bool = False,
    tissue_key: str = Key.tissue,
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

    Examples
    --------

    .. code::

        >>> from wsidata import open_wsi
        >>> import lazyslide as zs
        >>> wsi = open_wsi("https://github.com/camicroscope/Distro/raw/master/images/sample.svs")
        >>> zs.pp.find_tissues(wsi)
        >>> zs.pp.score_tiles(wsi, ["redness", "brightness"])
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


def _tissue_instance2poly(tissue, downsample, xoff=0, yoff=0):
    shell = tissue.contour * downsample
    holes = [hole * downsample for hole in tissue.holes]
    tissue_poly = Polygon(shell, holes=holes)
    return translate(tissue_poly, xoff=xoff, yoff=yoff)


def _get_downsample(wsi, level):
    if level == 0:
        return 1
    else:
        return wsi.properties.level_downsample[level]
