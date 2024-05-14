from __future__ import annotations

import warnings
from numbers import Integral
from typing import Sequence, Callable

import cv2
import numpy as np
from numba import njit

from zs.cv.scorer.base import ScorerBase
from zs.wsi import WSI, TileSpec


def tiles(
    wsi: WSI,
    tile_px: int,
    stride_px: int = None,
    edge: bool = False,
    mpp: float = None,
    tolerance: float = 0.05,
    background_fraction: float = 0.3,
    min_pts: int = 3,
    flavor: str = "polygon-test",
    filter: Callable = None,
    errors: str = "raise",
    tissue_key: str = "tissue",
    key: str = "tiles",
):
    """
    Generate tiles from the tissue contours

    Parameters
    ----------
    wsi : WSI
        The whole slide image object
    tile_px : int, (int, int)
        The size of the tile, if tuple, (W, H)
    stride_px : int, (int, int)
        The stride of the tile, if tuple, (W, H)
    edge : bool
        Whether to include the edge tiles
    mpp : float
        The requested mpp of the tiles, if None, use the slide mpp
    tolerance : float
        The tolerance when matching the mpp
    background_fraction : float
        For flavor='mask',
        The fraction of background in the tile, if more than this, discard the tile
    min_pts : int
        For flavor='polygon-test',
        The minimum number of points of a rectangle tile that should be inside the tissue
        should be within [0, 4]
    flavor : str, {'polygon-test', 'mask'}
        The flavor of the tile generation, either 'polygon-test' or 'mask'
        - 'polygon-test': Use point polygon test to check if the tiles points are inside the contours
        - 'mask': Transform the contours and holes into binary mask and check the fraction of background
    filter : Callable
        A callable that takes in a image and return a boolean value
    errors : str
        The error handling strategy, either 'raise' or 'warn'
    tissue_key : str, default 'tissue'
        The key of the tissue contours
    key : str, default 'tiles'
        The key of the tiles

    """
    # Check if tissue contours are present
    if f"{tissue_key}_contours" not in wsi.sdata.shapes:
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
        mpp = wsi.metadata.mpp
    else:
        if wsi.metadata.mpp is not None:
            downsample = mpp / wsi.metadata.mpp

            lower_ds = downsample - tolerance
            upper_ds = downsample + tolerance
            if lower_ds < 1 < upper_ds:
                downsample = 1

            if downsample < 1:
                raise ValueError(
                    f"Cannot perform resize operation "
                    f"with reqeust mpp={mpp} on image"
                    f"mpp={wsi.metadata.mpp}, this will"
                    f"require up-scaling of image."
                )
            elif downsample == 1:
                ops_level = 0
            else:
                for ix, level_downsample in enumerate(wsi.metadata.level_downsample):
                    if lower_ds < level_downsample < upper_ds:
                        downsample = level_downsample
                        ops_level = ix
                else:
                    run_downsample = True
        else:
            msg = f"{wsi.file} does not contain MPP."
            if errors:
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
    contours = wsi.sdata.shapes[f"{tissue_key}_contours"]
    if f"{tissue_key}_holes" in wsi.sdata.shapes:
        holes = wsi.sdata.shapes[f"{tissue_key}_holes"]
    else:
        holes = []

    tile_coords = []
    tiles_tissue_id = []
    for _, row in contours.iterrows():
        tissue_id, cnt = row
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
        cnt = np.array(cnt.exterior.coords, dtype=np.float32)
        if len(holes) > 0:
            cnt_holes = holes[holes["tissue_id"] == tissue_id]
            cnt_holes = [
                np.array(h.exterior.coords, dtype=np.float32)
                for h in cnt_holes.geometry
            ]
        else:
            cnt_holes = []

        # ========= 1. Point in polygon test =========
        if flavor == "polygon-test":
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
        elif flavor == "mask":
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
            coords += (minx, miny)
        else:
            msg = f"Unknown flavor {flavor}, supported flavors are ['polygon-test', 'mask']"
            raise NotImplementedError(msg)

        tile_coords.extend(coords)
        tiles_tissue_id.extend([tissue_id] * len(coords))

    tile_spec = TileSpec(
        level=ops_level,
        downsample=downsample,
        mpp=mpp,
        height=ops_tile_h,
        width=ops_tile_w,
        ops_height=ops_tile_h,
        ops_width=ops_tile_w,
        tissue_name=tissue_key,
    )

    if len(tile_coords) == 0:
        warnings.warn(
            "No tiles are found. " "Did you set a tile size that is too large?"
        )
    else:
        tile_coords = np.array(tile_coords)
        tiles_tissue_id = np.array(tiles_tissue_id)
        if filter is not None:
            to_use = []
            for t in tile_coords:
                img = wsi.get_region(
                    t[0], t[1], ops_tile_w, ops_tile_h, level=ops_level
                )
                use = filter(img)
                to_use.append(use)
            tile_coords = tile_coords[to_use]
            tiles_tissue_id = tiles_tissue_id[to_use]
        wsi.add_tiles(tile_coords, key, tile_spec, data={"tissue_id": tiles_tissue_id})


def score_tiles(
    wsi: WSI,
    scorer: ScorerBase | Callable,
    score_name: str = None,
    key: str = "tiles",
):
    """
    Score the tiles

    Parameters
    ----------
    wsi : WSI
        The whole slide image object
    scorer : ScorerBase or Callable
        The scorer object or a callable that takes in a image and return a score
    score_name : str
        The name of the score
    key : str
        The key of the tiles

    """

    tiles_tb = wsi.sdata.points[key]
    if hasattr(tiles_tb, "compute"):
        tiles_tb = tiles_tb.compute()
    spec = TileSpec(**wsi.sdata.tables[f"{key}_spec"].uns["tile_spec"])

    # Get the score function and name
    if isinstance(scorer, ScorerBase):
        score_func = scorer.get_score
        score_name = scorer.name if score_name is None else score_name
    else:
        score_func = scorer
        if score_name is None:
            if hasattr(scorer, "__name__"):
                score_name = f"{scorer.__name__}_score"
            else:
                score_name = "unknown_score"

    # Score the tiles
    tiles = tiles_tb[["x", "y"]].values
    scores = []
    for tile in tiles:
        x, y = tile
        img = wsi.get_region(x, y, spec.ops_width, spec.ops_height, level=spec.level)
        score = score_func(img)
        scores.append(score)

    # Get other columns in tiles table that are not x, y
    data = {}
    for col in tiles_tb.columns:
        if col not in ["x", "y"]:
            data[col] = tiles_tb[col].values
    data[score_name] = scores
    wsi.add_tiles(tiles, key, spec, data=data)


@njit
def create_tiles(image_shape, tile_w, tile_h, stride_w=None, stride_h=None, edge=True):
    """Create the tiles, return coordination that comprise the tiles
        and the index of points for each rect

    Parameters
    ----------
    image_shape : (int, int)
        The (H, W) of the image
    tile_w, tile_h: int
        The width/height of tile
    stride_w, stride_h : int
        The width/height of stride when move to next tile
    edge : bool
        Whether to include the edge tiles

    Returns
    -------
    coordinates : np.ndarray (N, 2)
        The coordinates of the tiles, N is the number of tiles
    indices : np.ndarray (M, 4)
        The indices of the points for each rect, M is the number of rects

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
    """Return a binary array that indicate which tile should be left

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
