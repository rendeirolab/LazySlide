from __future__ import annotations

import warnings
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
from numbers import Integral
from typing import Sequence

import cv2
import numpy as np
import pandas as pd
from lazyslide._const import Key
from lazyslide._preprocess._utils import get_scorer, Scorer
from lazyslide._utils import default_pbar, chunker, find_stack_level
from numba import njit, prange
from wsidata import WSIData, TileSpec
from wsidata.io import add_tiles, update_shapes_data


def tile_tissues(
    wsi: WSIData,
    tile_px: int | (int, int),
    stride_px: int | (int, int) = None,
    edge: bool = False,
    mpp: float = None,
    slide_mpp: float = None,
    tolerance: float = 0.05,
    method: str = "mask",
    background_fraction: float = 0.3,
    min_pts: int = 3,
    errors: str = "raise",
    tissue_key: str = Key.tissue,
    key_added: str = Key.tiles,
):
    """
    Generate tiles within the tissue contours in the WSI.

    Parameters
    ----------
    wsi : :class:`WSIData <wsidata.WSIData>`
        The WSIData object to work on.
    tile_px : int, (int, int)
        The size of the tile, if tuple, (W, H).
    stride_px : int, (int, int), default: None
        The stride of the tile, if tuple, (W, H).
        If None, use the tile size.
    edge : bool, default: False
        Whether to include the edge tiles.
    mpp : float, default: None
        The requested mpp of the tiles, if None, use the slide mpp.
    slide_mpp : float, default: None
        This value will override the slide mpp.
    tolerance : float, default: 0.05
        The tolerable deviation when matching the mpp.
        If requested mpp does not match the naive mpp,
        resize operation is needed when generating the tiles.
    method : str, {'polygon-test', 'mask'}
        The flavor of the tile generation, either 'polygon-test' or 'mask':
        - 'polygon-test': Use point polygon test to check if the tiles points are inside the contours.
        - 'mask': Transform the contours and holes into binary mask and check the fraction of background.
    background_fraction : float, default: 0.3
        For :code:`method='mask'`,
        The fraction of background in the tile, if more than this, discard the tile.
    min_pts : int, default: 3
        For :code:`method='polygon-test'`,
        The minimum number of points of a rectangle tile that should be inside the tissue.
        Should be within :code:`[0, 4]`.
    errors : {'raise', 'warn'}, default: 'raise'
        The error handling strategy, either 'raise' or 'warn'.
    tissue_key : str, default 'tissue'
        The key of the tissue contours.
    key_added : str, default 'tiles'
        The key of the tiles.

    Examples
    --------

    .. plot::
        :context: close-figs

        >>> from wsidata import open_wsi
        >>> import lazyslide as zs
        >>> wsi = open_wsi("https://github.com/camicroscope/Distro/raw/master/images/sample.svs")
        >>> zs.pp.find_tissues(wsi)
        >>> zs.pp.tile_tissues(wsi, 256, mpp=0.5)
        >>> zs.pl.tiles(wsi, tissue_id=0, show_grid=True, show_point=False)

    """
    # Check if tissue contours are present
    if tissue_key not in wsi.shapes:
        msg = f"Contours for {tissue_key} not found. Run pp.find_tissue first."
        raise ValueError(msg)

    if isinstance(tile_px, Integral):
        tile_w, tile_h = (tile_px, tile_px)
    elif isinstance(tile_px, Sequence):
        tile_w, tile_h = (tile_px[0], tile_px[1])
    else:
        raise TypeError(
            f"input tile_px {tile_px} invalid. "
            f"Either (W, H), or a single integer for square tiles."
        )

    assert tile_w > 0 and tile_h > 0, "Tile size must be positive."

    if stride_px is None:
        stride_w, stride_h = tile_w, tile_h
    elif isinstance(stride_px, Integral):
        stride_w, stride_h = (stride_px, stride_px)
    elif isinstance(stride_px, Sequence):
        stride_w, stride_h = (stride_px[0], stride_px[1])
    else:
        raise TypeError(
            f"input stride {stride_px} invalid. " f"Either (W, H), or a single integer."
        )

    assert stride_w > 0 and stride_h > 0, "Stride size must be positive."

    ops_level = 0
    downsample = 1
    run_downsample = False
    if mpp is None:
        mpp = wsi.properties.mpp
    if slide_mpp is None:
        slide_mpp = wsi.properties.mpp

    if slide_mpp is not None:
        downsample = mpp / slide_mpp

        lower_ds = downsample - tolerance
        upper_ds = downsample + tolerance
        if lower_ds < 1 < upper_ds:
            downsample = 1

        if downsample < 1:
            raise ValueError(
                f"Cannot perform resize operation "
                f"with reqeust mpp={mpp} on image"
                f"mpp={slide_mpp}, this will"
                f"require up-scaling of image."
            )
        elif downsample == 1:
            ops_level = 0
        else:
            for ix, level_downsample in enumerate(wsi.properties.level_downsample):
                if lower_ds < level_downsample < upper_ds:
                    downsample = level_downsample
                    ops_level = ix
            else:
                run_downsample = True
    else:
        msg = f"{wsi.reader.file} does not contain MPP."
        if errors == "raise":
            raise ValueError(msg)
        else:
            warnings.warn(msg, UserWarning, stacklevel=find_stack_level())

    if run_downsample:
        ops_tile_w = int(tile_w * downsample)
        ops_tile_h = int(tile_h * downsample)
        ops_stride_w = int(stride_w * downsample)
        ops_stride_h = int(stride_h * downsample)
    else:
        ops_tile_w, ops_tile_h = tile_w, tile_h
        ops_stride_w, ops_stride_h = stride_w, stride_h

    # Get contours
    contours = wsi.shapes[tissue_key]

    tile_coords = []
    tiles_tissue_id = []
    for _, row in contours.iterrows():
        tissue_id = row["tissue_id"]
        cnt = row["geometry"]
        minx, miny, maxx, maxy = cnt.bounds
        height, width = (maxy - miny, maxx - minx)

        rect_coords, rect_indices = create_tiles(
            (height, width),
            ops_tile_w,
            ops_tile_h,
            ops_stride_w,
            ops_stride_h,
            edge=edge,
        )
        # Dtype must be float32 for cv2
        cnt_holes = [np.asarray(h.coords, dtype=np.float32) for h in cnt.interiors]
        cnt = np.asarray(cnt.exterior.coords, dtype=np.float32)

        # ========= 1. Point in polygon test =========
        if method == "polygon-test":
            # Shift the coordinates to the correct position
            points = rect_coords + (minx, miny)
            in_cnt = [
                cv2.pointPolygonTest(cnt, (float(x), float(y)), measureDist=False)
                for x, y in points
            ]
            # Check if the point is inside the holes, init an array with False
            in_holes = np.zeros(len(points), dtype=bool)
            for h in cnt_holes:
                in_hole = [
                    cv2.pointPolygonTest(h, (float(x), float(y)), measureDist=False)
                    for x, y in points
                ]
                # Bitwise OR to get the points that are inside the holes
                in_holes = in_holes | (np.array(in_hole) >= 0)
            # Check if the point is inside the contours but not inside the holes
            is_in = np.array(in_cnt) >= 0
            is_in = is_in & ~in_holes
            # The number of points for each tiles inside contours
            good_tiles = is_in[rect_indices].sum(axis=1) >= min_pts
            # Select only the top_left corner
            coords = rect_coords[rect_indices[good_tiles, 0]].copy()
        # ========== 2. Transform contours and holes into binary mask ==========
        elif method == "mask":
            mask = np.zeros((int(height), int(width)), dtype=np.uint8)
            # Shift contour to the correct position
            cnt = (cnt - (minx, miny)).astype(np.int32)
            cv2.fillPoly(mask, [cnt], 1)
            for h in cnt_holes:
                h = (h - (minx, miny)).astype(np.int32)
                cv2.fillPoly(mask, [h], 0)
            good_tiles = filter_tiles(
                mask, rect_coords, ops_tile_w, ops_tile_h, background_fraction
            )
            coords = rect_coords[good_tiles].copy().astype(np.float32)
        else:
            msg = f"Unknown method {method}, supported methods are ['polygon-test', 'mask']"
            raise NotImplementedError(msg)

        tile_coords.extend(coords + (minx, miny))
        tiles_tissue_id.extend([tissue_id] * len(coords))

    tile_spec = TileSpec(
        level=ops_level,
        downsample=downsample,
        mpp=mpp,
        height=int(tile_h),
        width=int(tile_w),
        stride_height=int(stride_h),
        stride_width=int(stride_w),
        raw_height=int(ops_tile_h),
        raw_width=int(ops_tile_w),
        raw_stride_height=int(ops_stride_h),
        raw_stride_width=int(ops_stride_w),
        tissue_name=tissue_key,
    )

    if len(tile_coords) == 0:
        warnings.warn(
            "No tiles are found. Did you set a tile size that is too large?",
            UserWarning,
            stacklevel=find_stack_level(),
        )
    else:
        tile_coords = np.array(tile_coords).astype(np.uint)
        tiles_tissue_id = np.array(tiles_tissue_id)
        add_tiles(
            wsi,
            key=key_added,
            xys=tile_coords,
            tile_spec=tile_spec,
            tissue_ids=tiles_tissue_id,
        )


def tiles_qc(
    wsi: WSIData,
    scorers: Scorer | Sequence[Scorer] = None,
    num_workers: int = 1,
    pbar: bool = True,
    tile_key: str = Key.tiles,
    qc_key: str = Key.tile_qc,
):
    """
    Score the tiles and filter the tiles based on their quality scores.

    Parameters
    ----------
    wsi : :class:`WSIData <wsidata.WSIData>`
        The WSIData object to work on.
    scorers : Scorer
        The scorer object or a callable that takes in an image and returns a score.
        You can also pass in a string:
        - 'focus': A FocusLite scorer that will score the focus of the image
        - 'contrast': A Contrast scorer that will score the contrast of the image
        - 'brightness': A Brightness scorer that will score the brightness of the image
        - 'redness': A SplitRGB scorer that will score the redness of the image
    num_workers : int, default: 1
        The number of workers to use.
    pbar : bool, default: True
        Whether to show the progress bar or not.
    tile_key : str, default: 'tiles'
        The key of the tiles in the :bdg-danger:`shapes` slot.
    qc_key : str, default: 'qc'
        The key in the tiles dataframe indicates if a tile passed qc.

    Returns
    -------
    The columns with scores and the key added to the spatial data object.

    Examples
    --------
    .. code-block:: python

        >>> from wsidata import open_wsi
        >>> import lazyslide as zs
        >>> wsi = open_wsi("https://github.com/camicroscope/Distro/raw/master/images/sample.svs")
        >>> zs.pp.find_tissues(wsi)
        >>> zs.pp.tile_tissues(wsi, 256, mpp=0.5)
        >>> zs.pp.tiles_qc(wsi, scorers=["focus", "contrast"])
        >>> wsi['tiles'].head(n=2)

    """

    compose_scorer = get_scorer(scorers)

    if tile_key not in wsi:
        raise ValueError(f"Tiles with key {tile_key} not found.")
    tiles_tb = wsi[tile_key]
    spec = wsi.tile_spec(tile_key)

    with default_pbar(disable=not pbar) as progress_bar:
        task = progress_bar.add_task("Scoring tiles", total=len(tiles_tb))
        scores = []
        qc = []
        tiles = tiles_tb[["x", "y"]].values

        if num_workers == 1:
            # Score the tiles
            for tile in tiles:
                x, y = tile
                img = wsi.read_region(
                    x, y, spec.raw_width, spec.raw_height, level=spec.level
                )
                result = compose_scorer(img)
                scores.append(result.scores)
                qc.append(result.qc)
                progress_bar.update(task, advance=1)
        else:
            with Manager() as manager:
                queue = manager.Queue()
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    chunks = chunker(tiles, num_workers)
                    wsi.reader.detach_reader()
                    futures = [
                        executor.submit(
                            _chunk_scoring,
                            chunk,
                            spec,
                            wsi.reader,
                            compose_scorer,
                            queue,
                        )
                        for chunk in chunks
                    ]
                    while any(future.running() for future in futures):
                        if queue.empty():
                            continue
                        _ = queue.get()
                        progress_bar.update(task, advance=1)
                    for f in futures:
                        chunk_scores, chunk_qc = f.result()
                        scores.extend(chunk_scores)
                        qc.extend(chunk_qc)
        progress_bar.refresh()

    scores = pd.DataFrame(scores).assign(**{qc_key: qc})
    update_shapes_data(wsi, key=tile_key, data=scores)


def _chunk_scoring(tiles, spec, reader, scorer, queue):
    scores = []
    qc = []
    for tile in tiles:
        x, y = tile
        img = reader.get_region(x, y, spec.raw_width, spec.raw_height)
        result = scorer(img)
        scores.append(result.scores)
        qc.append(result.qc)
        queue.put(1)
    return scores, qc


@njit
def create_tiles(
    image_shape, tile_w: int, tile_h: int, stride_w=None, stride_h=None, edge=True
):
    """Create the tiles, return coordination that comprise the tiles
        and the index of points for each rectangular.

    Tips: The number of tiles has nothing to do with the tile size,
    it's decided by stride size.

    Parameters
    ----------
    image_shape : (int, int)
        The (H, W) of the image.
    tile_w, tile_h: int
        The width/height of tiles.
    stride_w, stride_h : int, default None
        The width/height of stride when moving to the next tile.
    edge : bool, default True
        Whether to include the edge tiles.

    Returns
    -------
    coordinates : np.ndarray (N, 2)
        The coordinates of the tiles, N is the number of tiles.
    indices : np.ndarray (M, 4)
        The indices of the points for each rect, M is the number of rects.

    """
    height, width = image_shape

    if stride_w is None:
        stride_w = tile_w
    if stride_h is None:
        stride_h = tile_h

    # calculate number of expected tiles
    # If the width/height is divisible by stride
    # We need to add 1 to include the starting point
    nw = width // stride_w + 1
    nh = height // stride_h + 1

    # To include the edge tiles
    if edge and width % stride_w != 0:
        nw += 1
    if edge and height % stride_h != 0:
        nh += 1

    coordinates = list()
    indices = list()

    xs = np.arange(nw, dtype=np.uint) * stride_w
    ys = np.arange(nh, dtype=np.uint) * stride_h

    # Filter out the tiles that are out of the image
    xs = xs[xs < width]
    ys = ys[ys < height]
    # Update the number of tiles
    nw = len(xs)
    nh = len(ys)

    xv, yv = meshgrid(xs, ys)

    for i in range(nw):
        for j in range(nh):
            coordinates.append([xv[j, i], yv[j, i]])

    if nw == 1 and nh == 1:
        n_rect = 1
    else:
        n_rect = (nw - 1) * (nh - 1)
    s1, s2, s3, s4 = 0, 1, nh + 1, nh
    for i in range(n_rect):
        indices.append([s1 + i, s2 + i, s3 + i, s4 + i])

    return np.array(coordinates, dtype=np.uint), np.array(indices, dtype=np.uint)


@njit
def meshgrid(x, y):
    nx = x.size
    ny = y.size

    X = np.zeros((ny, nx), dtype=np.uint)
    Y = np.zeros((ny, nx), dtype=np.uint)

    for i in range(ny):
        for j in range(nx):
            X[i, j] = x[j]
            Y[i, j] = y[i]

    return X, Y


@njit
def filter_tiles(mask, tiles_coords, tile_w, tile_h, filter_bg=0.8):
    """Returns a binary array that indicate which tile should be left.

    Parameters
    ----------
    mask
    tiles_coords
    filter_bg
    tile_w,
    tile_h,
    Returns
    -------

    """
    n = len(tiles_coords)
    use = np.zeros(n, dtype=np.bool_)
    mask_h, mask_w = mask.shape
    for i in prange(n):
        w, h = tiles_coords[i]
        h_end, w_end = min(h + tile_h, mask_h), min(w + tile_w, mask_w)
        sub_mask = mask[h:h_end, w:w_end]
        if sub_mask.shape != (tile_h, tile_w):
            # Padding with 0
            mask_region = np.zeros((tile_h, tile_w), dtype=np.uint8)
            mask_region[: h_end - h, : w_end - w] = sub_mask
        else:
            mask_region = sub_mask
        # Both 0 and 255 are considered as background
        bg_ratio = np.sum(mask_region == 0) / mask_region.size
        use[i] = bg_ratio < filter_bg
    return use
