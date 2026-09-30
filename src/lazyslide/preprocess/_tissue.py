from __future__ import annotations

import logging
import math
import warnings
from typing import Literal

import cv2
import geopandas as gpd
import numpy as np
import psutil
from shapely.affinity import scale, translate
from wsidata import WSIData
from wsidata.io import add_tissues

from lazyslide.cv.mask import BinaryMask
from lazyslide.cv.transform import (
    ArtifactFilterThreshold,
    BinaryThreshold,
    Compose,
    EntropyThreshold,
    MedianBlur,
    MorphClose,
)

from .._const import Key
from .._utils import find_stack_level
from ..cv import merge_connected_polygons

logger = logging.getLogger(__name__)


def _tissue_mask(
    image,
    to_hsv,
    filter_artifacts: bool = True,
    blur_ksize: int = 7,
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


def _build_tissue_mask(image, method, otsu_kwargs, entropy_kwargs):
    if method == "otsu":
        return _tissue_mask(image, **otsu_kwargs)
    if method == "entropy":
        kwargs = dict(entropy_kwargs)
        morph_ksize = kwargs.pop("morph_ksize")
        morph_n_iter = kwargs.pop("morph_n_iter")
        c = Compose(
            [
                EntropyThreshold(**kwargs),
                MorphClose(kernel_size=morph_ksize, n_iterations=morph_n_iter),
            ]
        )
        return c.apply(image)
    raise ValueError(f"Unknown method: {method!r}. Choose from 'otsu' or 'entropy'.")


def find_tissues(
    wsi: WSIData,
    level: int | str = "auto",
    refine_level: int | str | None = None,
    method: Literal["otsu", "entropy"] = "otsu",
    to_hsv: bool = False,
    blur_ksize: int = 7,
    threshold: int = 7,
    morph_n_iter: int = 3,
    morph_ksize: int = 7,
    min_tissue_area: float = 1e-3,
    min_hole_area: float = 1e-5,
    detect_holes: bool = True,
    filter_artifacts: bool = True,
    disk_radius: int = 4,
    relaxed_threshold: bool = True,
    invert_check: bool = True,
    in_bounds: bool = True,
    key_added: str = Key.tissue,
):
    """Find tissue regions in the :term:`WSI` and add them as :term:`contours` and :term:`holes`.

    .. note::
        The results may not be deterministic between runs,
        as the :term:`segmentation level` is automatically decided by the available memory.
        To get a consistent result, you can set the `level` parameter to a specific value.
        Set `level=-1` for the lowest resolution level and fastest :term:`tissue segmentation` speed.

    .. seealso::
        :func:`zs.seg.tissue <lazyslide.seg.tissue>`


    Parameters
    ----------
    wsi : :class:`WSIData <wsidata.WSIData>`
        The :term:`WSIData` object to work on.
    level : int, default: 'auto'
        The level to use for segmentation.
    refine_level : int or 'auto', default: None
        The level to refine the tissue polygons.
    method : {'otsu', 'entropy'}, default: 'otsu'
        Tissue mask construction strategy.
        ``'otsu'`` uses median blur + Otsu/artifact thresholding + morphological
        closing. ``'entropy'`` uses HED color-space local entropy with Otsu
        thresholding followed by morphological closing; when
        ``method='entropy'`` the otsu-specific parameters (``to_hsv``,
        ``blur_ksize``, ``threshold``, ``filter_artifacts``) are ignored.
    to_hsv : bool, default: False
        (otsu only) The tissue image will be converted from RGB to HSV space,
        the saturation channel (color purity) will be used for tissue detection.
    blur_ksize : int, default: 7
        (otsu only) The kernel size used to apply median blurring.

        .. note::
            This default changed from 17 to 7. Previously a shared mutable dict
            on ``Transform`` meant the blur silently ran with ``morph_ksize``
            (7) rather than ``blur_ksize``, so 17 was never the kernel actually
            used. Now that the two are independent, defaulting to 7 keeps the
            output of a default call unchanged. Pass ``blur_ksize=17`` for the
            blur the old signature advertised.
    threshold : int, default: 7
        (otsu only) The threshold for binary thresholding.
    morph_n_iter : int, default: 3
        The number of iterations of morphological closing to apply
        (also applied as opening on the otsu path).
    morph_ksize : int, default: 7
        The kernel size for morphological closing
        (also applied as opening on the otsu path).
    min_tissue_area : float, default: 1e-3
        The minimum area of tissue.
    min_hole_area : float, default: 1e-5
        The minimum area of holes.
    detect_holes : bool, default: True
        Detect holes in tissue regions.
    filter_artifacts : bool, default: True
        (otsu only) Filter :term:`artifacts <artifact>` out. Artifacts that are non-redish are removed.
    disk_radius : int, default: 4
        (entropy only) Radius of the disk structuring element used for entropy filtering.
    relaxed_threshold : bool, default: True
        (entropy only) If True, use a more permissive threshold (0.85x Otsu)
        and aggregate only hematoxylin and eosin entropy. If False, also
        subtract DAB entropy, producing a stricter mask.
    invert_check : bool, default: True
        (entropy only) Whether to detect and correct mask inversion when the
        background dominates the image borders.
    in_bounds : bool, default: True
        Only segment the region inside the slide bounds, e.g. the scanned area
        of MRXS slides. This reads less of the background, takes less memory,
        and lets ``level='auto'`` choose a finer level. Slides without bounds
        are segmented as a whole.
    key_added : str, default: 'tissues'
        The key to save the result in the :term:`WSIData` object.

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

    ops_level = _decide_level(wsi, level, proportion=proportion, in_bounds=in_bounds)
    # Set the segmentation options

    # Run the first segmentation
    otsu_kwargs = {
        "to_hsv": to_hsv,
        "filter_artifacts": filter_artifacts,
        "blur_ksize": blur_ksize,
        "threshold": threshold,
        "morph_ksize": morph_ksize,
        "morph_n_iter": morph_n_iter,
    }
    entropy_kwargs = {
        "disk_radius": disk_radius,
        "relaxed_threshold": relaxed_threshold,
        "invert_check": invert_check,
        "morph_ksize": morph_ksize,
        "morph_n_iter": morph_n_iter,
    }
    to_poly_option = {
        "min_area": min_tissue_area,
        "min_hole_area": min_hole_area,
    }
    tissue_image = wsi.reader.get_level(ops_level, in_bounds=in_bounds)
    tissue_mask = _build_tissue_mask(tissue_image, method, otsu_kwargs, entropy_kwargs)
    tissue_polys = BinaryMask(tissue_mask).to_polygons(
        **to_poly_option, detect_holes=detect_holes_1
    )
    tissue_polys = tissue_polys.geometry

    if len(tissue_polys) == 0:
        logger.warning("No tissue is found.", stacklevel=find_stack_level())
        return False

    tissues = []
    downsample = _get_downsample(wsi, ops_level)
    # The image starts at the bounds origin
    x0, y0, _, _ = _slide_region(wsi.properties, in_bounds)
    for tissue in tissue_polys:
        # Scale it back to level 0
        tissue = scale(tissue, xfact=downsample, yfact=downsample, origin=(0, 0))
        tissues.append(translate(tissue, xoff=x0, yoff=y0))

    if refine_level is not None:
        # Refine the tissue polygons at a higher resolution level
        refine_tissues = []
        for tissue_poly in tissues:
            # Tissue polygon at the highest resolution level
            xmin, ymin, xmax, ymax = tissue_poly.bounds
            # Enlarge the bounding box by 10%
            width, height = xmax - xmin, ymax - ymin

            if refine_level == "auto":
                current_refine_level = _decide_level(
                    wsi, refine_level, proportion=proportion, in_bounds=in_bounds
                )
                if current_refine_level == ops_level:
                    current_refine_level -= 1
                current_refine_level = max(current_refine_level, 0)

            else:
                current_refine_level = refine_level

            refine_downsample = _get_downsample(wsi, current_refine_level)

            image = _read_region(wsi, xmin, ymin, width, height, current_refine_level)
            tissue_mask = _build_tissue_mask(image, method, otsu_kwargs, entropy_kwargs)
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
    region = _slide_region(metadata, in_bounds)
    width, height = _size_at_level(metadata, level, *region)
    # The data type in uint8, so each pixel is 1 byte
    # The size is calculated by width * height * 4 (RGBA)
    bytes_size = width * height * 4
    # if the size is beyond 4GB, use a higher level
    while bytes_size > available_memory:
        if level != metadata.n_level - 1:
            level += 1
            width, height = _size_at_level(metadata, level, *region)
            bytes_size = width * height * 4
        else:
            level = metadata.n_level - 1
            break
    if warn:
        warnings.warn(f"mpp is not available, use level {level} for segmentation.")
    return level


def _decide_level(wsi, level, proportion=0.8, in_bounds=True):
    if level == "auto":
        return _get_optimal_level(
            wsi.properties, in_bounds=in_bounds, proportion=proportion
        )
    else:
        return wsi.reader.translate_level(level)


def _slide_region(properties, in_bounds):
    """The level-0 (x, y, width, height) that ``get_level`` reads."""
    if in_bounds:
        return tuple(properties.bounds)
    height, width = properties.level_shape[0]
    return 0, 0, width, height


def _size_at_level(properties, level, x, y, width, height):
    """The (width, height) at ``level`` of a level-0 region, clipped to the level.

    Reading past the level edge returns transparent pixels, which turn black.
    """
    ds = 1 if level == 0 else properties.level_downsample[level]
    level_height, level_width = properties.level_shape[level]
    return (
        min(math.ceil(width / ds), level_width - int(x / ds)),
        min(math.ceil(height / ds), level_height - int(y / ds)),
    )


def _read_region(wsi, x, y, width, height, level):
    """Read a level-0 region at ``level``; the reader takes the size at that level."""
    w, h = _size_at_level(wsi.properties, level, x, y, width, height)
    return wsi.reader.get_region(x, y, w, h, level=level)


def _get_downsample(wsi, level):
    if level == 0:
        return 1
    else:
        return wsi.properties.level_downsample[level]
