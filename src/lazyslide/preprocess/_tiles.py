from __future__ import annotations

import warnings
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
from numbers import Integral
from typing import Sequence, Callable

import cv2
import numpy as np
import pandas as pd
from numba import njit

from lazyslide._const import Key
from lazyslide.preprocess._utils import get_scorer, Scorer
from lazyslide._utils import default_pbar, chunker
from wsidata import WSIData, TileSpec


def tiles(
    wsi: WSIData,
    tile_px: int,
    stride_px: int = None,
    edge: bool = False,
    mpp: float = None,
    slide_mpp: float = None,
    tolerance: float = 0.05,
    background_fraction: float = 0.3,
    min_pts: int = 3,
    method: str = "mask",
    filter: Callable = None,
    errors: str = "raise",
    tissue_key: str = Key.tissue,
    key_added: str = Key.tiles,
):
    """
    Generate tiles within the tissue contours in the WSI.

    Parameters
    ----------
    wsi : WSI
        The whole slide image object.
    tile_px : int, (int, int)
        The size of the tile, if tuple, (W, H).
    stride_px : int, (int, int), default None
        The stride of the tile, if tuple, (W, H).
    edge : bool, default False
        Whether to include the edge tiles.
    mpp : float, default None
        The requested mpp of the tiles, if None, use the slide mpp.
    slide_mpp : float, default None
        This value will override the slide mpp.
    tolerance : float, default 0.05
        The tolerance when matching the mpp.
    background_fraction : float, default 0.3
        For flavor='mask',
        The fraction of background in the tile, if more than this, discard the tile.
    min_pts : int
        For flavor='polygon-test',
        The minimum number of points of a rectangle tile that should be inside the tissue.
        Should be within [0, 4].
    method : str, {'polygon-test', 'mask'}
        The flavor of the tile generation, either 'polygon-test' or 'mask':
        - 'polygon-test': Use point polygon test to check if the tiles points are inside the contours.
        - 'mask': Transform the contours and holes into binary mask and check the fraction of background.
    filter : Callable, default None
        A callable that takes in a image and return a boolean value.
    errors : str, default 'raise'
        The error handling strategy, either 'raise' or 'warn'.
    tissue_key : str, default 'tissue'
        The key of the tissue contours.
    key_added : str, default 'tiles'
        The key of the tiles.

    Examples
    --------

    .. plot::
        :context: close-figs

        >>> import lazyslide as zs
        >>> wsi = zs.open_wsi("https://github.com/camicroscope/Distro/raw/master/images/sample.svs")
        >>> zs.preprocess.find_tissue(wsi)
        >>> zs.preprocess.tiles(wsi, 256, mpp=0.5)
        >>> zs.plotting.tiles(wsi, tissue_id=0, show_grid=True, show_point=False)

    """
    # Check if tissue contours are present
    if tissue_key not in wsi.sdata.shapes:
        msg = f"Contours for {tissue_key} not found. Run preprocess.find_tissue first."
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

    if stride_px is None:
        stride_h, stride_w = tile_h, tile_w
    elif isinstance(stride_px, Integral):
        stride_h, stride_w = (stride_px, stride_px)
    elif isinstance(stride_px, Sequence):
        stride_h, stride_w = (stride_px[0], stride_px[1])
    else:
        raise TypeError(
            f"input stride {stride_px} invalid. " f"Either (W, H), or a single integer."
        )

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
            warnings.warn(msg)

    if run_downsample:
        ops_tile_w = int(tile_w * downsample)
        ops_tile_h = int(tile_h * downsample)
        ops_stride_w = int(stride_w * downsample)
        ops_stride_h = int(stride_h * downsample)
    else:
        ops_tile_w, ops_tile_h = tile_w, tile_h
        ops_stride_w, ops_stride_h = stride_w, stride_h

    # Get contours
    contours = wsi.sdata.shapes[tissue_key]

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
        cnt_holes = [np.asarray(h.coords) for h in cnt.interiors]
        cnt = np.asarray(cnt.exterior.coords, dtype=np.float32)

        # ========= 1. Point in polygon test =========
        if method == "polygon-test":
            points = rect_coords + (minx, miny)
            # Shift the coordinates to the correct position
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
        height=ops_tile_h,
        width=ops_tile_w,
        raw_height=ops_tile_h,
        raw_width=ops_tile_w,
        tissue_name=tissue_key,
    )

    if len(tile_coords) == 0:
        warnings.warn(
            "No tiles are found. " "Did you set a tile size that is too large?"
        )
    else:
        tile_coords = np.array(tile_coords).astype(np.uint)
        tiles_tissue_id = np.array(tiles_tissue_id)
        if filter is not None:
            to_use = []
            for t in tile_coords:
                img = wsi.read_region(
                    t[0], t[1], ops_tile_w, ops_tile_h, level=ops_level
                )
                use = filter(img)
                to_use.append(use)
            tile_coords = tile_coords[to_use]
            tiles_tissue_id = tiles_tissue_id[to_use]
        wsi.add_tiles(
            key=key_added,
            xys=tile_coords,
            tile_spec=tile_spec,
            tissue_ids=tiles_tissue_id,
        )


def tiles_qc(
    wsi: WSIData,
    scorers: Scorer | Sequence[Scorer],
    num_workers: int = 1,
    pbar: bool = True,
    key: str = Key.tiles,
    qc_key: str = Key.tile_qc,
):
    """
    Score the tiles and filter the tiles based on their quality scores.

    Parameters
    ----------
    wsi : WSI
        The whole slide image object.
    scorers : ScorerBase
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
    key : str
        The key of the tiles.

    Examples
    --------
    .. code-block:: python

        >>> import lazyslide as zs
        >>> wsi = zs.open_wsi("https://github.com/camicroscope/Distro/raw/master/images/sample.svs")
        >>> zs.preprocess.find_tissue(wsi)
        >>> zs.preprocess.tiles(wsi, 256, mpp=0.5)
        >>> zs.preprocess.tiles_qc(wsi, scorers=["contrast"])
        >>> wsi.sdata['tiles'].head(n=2)

    """

    compose_scorer = get_scorer(scorers)

    if key not in wsi.sdata:
        raise ValueError(f"Tile {key} not found.")
    tiles_tb = wsi.sdata[key]
    spec = wsi.tile_spec(key)

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
    wsi.update_shapes_data(key, scores)


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
def create_tiles(image_shape, tile_w, tile_h, stride_w=None, stride_h=None, edge=True):
    """Create the tiles, return coordination that comprise the tiles
        and the index of points for each rectangular.

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
    if edge and width % stride_w != 0:
        nw = width // stride_w + 1
    else:
        nw = (width - tile_w) // stride_w + 1
    if edge and height % stride_h != 0:
        nh = height // stride_h + 1
    else:
        nh = (height - tile_h) // stride_h + 1
    xs = np.arange(0, nw) * stride_w
    ys = np.arange(0, nh) * stride_h

    coordinates = list()
    indices = list()

    track_ix = 0
    for x in xs:
        for y in ys:
            r1 = (x, y)
            r2 = (x + 2, y)
            r3 = (x + 2, y + 2)
            r4 = (x, y + 2)
            coordinates.append(r1)
            coordinates.append(r2)
            coordinates.append(r3)
            coordinates.append(r4)
            indices.append([track_ix, track_ix + 1, track_ix + 2, track_ix + 3])
            track_ix += 4

    return np.array(coordinates, dtype=np.uint), np.array(indices, dtype=np.uint)


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
    use = []
    mask_h, mask_w = mask.shape
    for w, h in tiles_coords:
        h_end, w_end = min(h + tile_h, mask_h), min(w + tile_w, mask_w)
        mask_region = mask[h:h_end, w:w_end]
        bg_ratio = np.sum(mask_region == 0) / mask_region.size
        use.append(bg_ratio < filter_bg)
    return np.array(use, dtype=np.bool_)