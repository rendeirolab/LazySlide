from __future__ import annotations

import warnings
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
from typing import Literal, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import Polygon, box, contains_xy, prepare
from spatialdata.models import ShapesModel
from wsidata import TileSpec, WSIData
from wsidata.io import update_shapes_data
from wsidata.reader import ReaderBase

from lazyslide._const import Key
from lazyslide._utils import chunker, default_pbar, find_stack_level
from lazyslide.preprocess._utils import Scorer, get_scorer


def tile_tissues(
    wsi: WSIData,
    tile_px: int | (int, int),
    *,
    stride_px: int | (int, int) = None,
    overlap: float = None,
    edge: bool = False,
    mpp: float = None,
    slide_mpp: float = None,
    ops_level: int = None,
    background_filter: bool = True,
    background_fraction: float = 0.3,
    background_filter_mode: Literal["approx", "exact"] = "approx",
    tissue_key: str | None = Key.tissue,
    key_added: str | None = Key.tiles,
    return_tiles: bool = False,
):
    """
    Generate tiles within the tissue contours in the WSI.

    If there is no tissue contours, the tiles will generate for the whole image.

    Parameters
    ----------
    wsi : :class:`WSIData <wsidata.WSIData>`
        The WSIData object to work on.
    tile_px : int, (int, int)
        The size of the tile, if tuple, (W, H).
    stride_px : int, (int, int), default: None
        The stride of the tile, if tuple, (W, H).
        If None, use the tile size.
    overlap : float, default: None
        The overlap of the tiles, exclusive with stride_px.
        If in (0, 1), it's the overlap ratio.
        If > 1, it's the overlap in pixels.
    edge : bool, default: False
        Whether to include the edge tiles.
    mpp : float, default: None
        The requested mpp of the tiles, if None, use the slide mpp.
    slide_mpp : float, default: None
        This value will override the slide mpp.
    ops_level : int, default: None
        Which level to use for the actual tile image retrival.
    background_filter : bool, default: True
        Whether to filter the tiles based on the background fraction.
    background_fraction : float, default: 0.3
        Only used if `background_filter` is True.
        The fraction of background in the tile, if more than this, discard the tile.
    background_filter_mode : {'approx', 'exact'}, default: 'approx'
        The mode of the background filter.
        If 'approx', it will filter out tiles that are on the border of the tissue.
        If 'exact', it will filter out tiles that are not within the tissue,
        but it may be much slower for smaller tiles.
    tissue_key : str, default 'tissue'
        The key of the tissue contours.
    key_added : str, default 'tiles'
        The key of the tiles. If set to None, the tiles will not be added to the WSIData object.
    return_tiles : bool, default: False
        Return the tiles dataframe.

    Returns
    -------
    :class:`geopandas.GeoDataFrame`
        The tiles dataframe with columns :code:`tissue_id`, :code:`tile_id`, and :code:`geometry`.\
        Added to :bdg-danger:`shapes`.
    :class:`TileSpec <wsidata.TileSpec>`
        The tile specification used to create the tiles.\
        Added to :bdg-danger:`attrs` with key :code:`tile_spec`.

    Examples
    --------

    .. plot::
        :context: close-figs

        >>> import lazyslide as zs
        >>> wsi = zs.datasets.sample()
        >>> zs.pp.find_tissues(wsi)
        >>> zs.pp.tile_tissues(wsi, 256, mpp=0.5)
        >>> zs.pl.tiles(wsi, linewidth=0.5)

    """

    # Check if tissue contours are present
    if tissue_key not in wsi.shapes:
        msg = f"Contours for {tissue_key} not found. Consider run pp.find_tissue first."
        warnings.warn(msg, stacklevel=find_stack_level())
        tissue_key = None

    # Create the tile spec
    tile_spec = TileSpec.from_wsidata(
        wsi,
        tile_px=tile_px,
        stride_px=stride_px,
        overlap=overlap,
        mpp=mpp,
        ops_level=ops_level,
        slide_mpp=slide_mpp,
        tissue_name=tissue_key,
    )

    # Get contours
    if tissue_key is not None:
        contours = wsi.shapes[tissue_key]
    else:
        # If no tissue contours, use the whole slide image
        x, y, w, h = wsi.properties.bounds
        tiles = tiles_from_bbox(
            x,
            y,
            w,
            h,
            tile_spec.base_width,
            tile_spec.base_height,
            stride_w=tile_spec.base_stride_width,
            stride_h=tile_spec.base_stride_height,
            edge=edge,
        )
        _add_tiles(wsi, tiles, tile_spec, key_added=key_added)
        if return_tiles:
            return tiles, tile_spec
        return None

    # tile_coords = []
    tiles_collections = []
    tiles_tissue_id = []
    for _, row in contours.iterrows():
        tissue_id = row["tissue_id"]
        cnt = row["geometry"]
        minx, miny, maxx, maxy = cnt.bounds
        height, width = (maxy - miny, maxx - minx)

        tiles = tiles_from_bbox(
            minx,
            miny,
            width,
            height,
            tile_spec.base_width,
            tile_spec.base_height,
            stride_w=tile_spec.base_stride_width,
            stride_h=tile_spec.base_stride_height,
            edge=edge,
            mask=cnt,
        )

        if background_filter:
            if background_filter_mode == "approx":
                # check for tiles that are on the border of the tissue
                border_tiles = tiles[tiles["pt_count"] < 4]
            elif background_filter_mode == "exact":
                border_tiles = tiles
                if len(border_tiles) > 5000:
                    # If there are too many tiles, warn the user
                    # This is to avoid performance issues with exact mode
                    warnings.warn(
                        "Using 'exact' mode for background_filter_mode may be computationally expensive "
                        "for large numbers of tiles. Consider using 'approx' mode for better performance.",
                        stacklevel=find_stack_level(),
                    )
            else:
                raise ValueError(
                    f"Unknown background filter mode: {background_filter_mode}. "
                    "Use 'approx' or 'exact'."
                )
            # calculate the background fraction of each tile
            ov_ratio = border_tiles.intersection(cnt).area / border_tiles.area

            # filter out the tiles that are not within the tissue
            exclude_tiles = ov_ratio[ov_ratio < (1 - background_fraction)]
            overlap_tiles = tiles.drop(index=exclude_tiles.index, errors="ignore")

        else:
            # If no background filter, just use the intersecting tiles
            overlap_tiles = tiles.drop(columns="index_right", errors="ignore")
            # overlap_tiles = tiles

        # Add to the final collection and match the tissue id
        tiles_collections.append(overlap_tiles)
        tiles_tissue_id.extend([tissue_id] * len(overlap_tiles))

    tiles = pd.concat(tiles_collections).reset_index(drop=True)
    if len(tiles) == 0:
        warnings.warn(
            "No tiles are found. Did you set a tile size that is too large?",
            UserWarning,
            stacklevel=find_stack_level(),
        )
        return None
    else:
        tiles["tissue_id"] = tiles_tissue_id
        tiles["tile_id"] = np.arange(len(tiles))
        # reorganize the columns
        tiles = tiles[["tile_id", "tissue_id", "geometry"]]
        _add_tiles(wsi, tiles, tile_spec, key_added=key_added)
        if return_tiles:
            return tiles, tile_spec
        return None


def _add_tiles(wsi, tiles_gdf, tile_spec, key_added=None):
    if key_added is None:
        return

    wsi.shapes[key_added] = ShapesModel.parse(tiles_gdf)

    if wsi.TILE_SPEC_KEY in wsi.attrs:
        spec_data = wsi.attrs[wsi.TILE_SPEC_KEY]
        spec_data[key_added] = tile_spec.to_dict()
    else:
        spec_data = {key_added: tile_spec.to_dict()}
        wsi.attrs[wsi.TILE_SPEC_KEY] = spec_data


def score_tiles(
    wsi: WSIData,
    scorers: Scorer | Sequence[Scorer] = None,
    num_workers: int = 1,
    pbar: bool = True,
    tile_key: str = Key.tiles,
):
    """
    Score the tiles and filter the tiles based on their quality scores.

    .. deprecated:: 0.7.2

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

    Returns
    -------
    None

    .. note::
        The scores will be added to the :code:`tiles | {tile_key}` table in the WSIData object.
        The columns will be named after the scorers, e.g. `redness`, `brightness`.

    Examples
    --------
    .. plot::
        :context: close-figs

        >>> import lazyslide as zs
        >>> wsi = zs.datasets.sample()
        >>> zs.pp.find_tissues(wsi)
        >>> zs.pp.tile_tissues(wsi, 256, mpp=0.5)
        >>> zs.pp.score_tiles(wsi, scorers=["focus", "contrast"])
        >>> zs.pl.tiles(wsi, color=["focus", "contrast"])

    """

    warnings.warn(
        "This function is deprecated and will be removed after v0.9.0, "
        "please use `tl.tile_prediction` instead.",
        stacklevel=find_stack_level(),
    )

    compose_scorer = get_scorer(scorers)

    if tile_key not in wsi:
        raise ValueError(f"Tiles with key {tile_key} not found.")
    tiles_tb = wsi[tile_key]
    spec = wsi.tile_spec(tile_key)

    with default_pbar(disable=not pbar) as progress_bar:
        task = progress_bar.add_task("Scoring tiles", total=len(tiles_tb))
        scores = []
        qc = []
        tiles = tiles_tb.bounds[["minx", "miny"]].values

        if num_workers == 1:
            # Score the tiles
            for tile in tiles:
                x, y = tile
                img = wsi.read_region(
                    x, y, spec.ops_width, spec.ops_height, level=spec.ops_level
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

    scores = pd.DataFrame(scores)  # .assign(**{qc_key: qc})
    update_shapes_data(wsi, key=tile_key, data=scores)


def _chunk_scoring(tiles, spec: TileSpec, reader: ReaderBase, scorer, queue):
    scores = []
    qc = []
    for tile in tiles:
        x, y = tile
        img = reader.get_region(
            x, y, spec.ops_width, spec.ops_height, level=spec.ops_level
        )
        result = scorer(img)
        scores.append(result.scores)
        qc.append(result.qc)
        queue.put(1)
    return scores, qc


def tiles_from_bbox(
    x,
    y,
    w,
    h,
    tile_w: int,
    tile_h: int,
    stride_w=None,
    stride_h=None,
    edge=True,
    mask=None,
):
    """Create tiles from a bounding box.

    Parameters
    ----------
    x, y, w, h : int
        The x, y, width, height of the bounding box.
    tile_w, tile_h: int
        The width/height of tiles.
    stride_w, stride_h : int, default None
        The width/height of stride when moving to the next tile.
    edge : bool, default True
        Whether to include the edge tiles.
    mask : Polygon, default None
        The mask to use for the tiles.

    Returns
    -------
    List[Polygon]
        The list of tiles.

    """
    # A new implementation in pure numpy and return shapely geometry
    x, y, w, h = int(x), int(y), int(w), int(h)

    if stride_w is None:
        stride_w = tile_w
    if stride_h is None:
        stride_h = tile_h

    # calculate the number of expected tiles
    # If the width/height is divisible by stride,
    # We need to add 1 to include the starting point
    nw = w // stride_w + 1
    nh = h // stride_h + 1

    # To include the edge tiles
    if edge and w % stride_w != 0:
        nw += 1
    if edge and h % stride_h != 0:
        nh += 1

    xs = np.arange(nw, dtype=np.uint) * stride_w + x
    ys = np.arange(nh, dtype=np.uint) * stride_h + y
    # points = np.array(np.meshgrid(xs, ys)).T.reshape(-1, 2)

    # Add xs and ys after stride
    if stride_h != tile_h:
        yss = ys + tile_h
        yss = np.sort(np.unique(np.concatenate((ys, yss))))
    else:
        yss = ys
    if stride_w != tile_w:
        xss = xs + tile_w
        xss = np.sort(np.unique(np.concatenate((xs, xss))))
    else:
        xss = xs

    tiles = []
    pt_counts = []
    if mask is not None:
        # Filter the points that are within the mask
        tile_points = np.array(np.meshgrid(xss, yss)).T.reshape(-1, 2)
        prepare(mask)
        is_in = contains_xy(mask, x=tile_points[:, 0], y=tile_points[:, 1])
        # make a dict mapping if the point is in the mask
        in_dict = dict(zip(map(tuple, tile_points), is_in))
        for i in range(nw):
            for j in range(nh):
                x, y = xs[i], ys[j]
                p1, p2, p3, p4 = (
                    (x, y),
                    (x + tile_w, y),
                    (x + tile_w, y + tile_h),
                    (x, y + tile_h),
                )
                pt_count = sum(in_dict.get(p, 0) for p in (p1, p2, p3, p4))
                if pt_count > 0:
                    tiles.append(box(x, y, x + tile_w, y + tile_h))
                    pt_counts.append(pt_count)
    else:
        for i in range(nw):
            for j in range(nh):
                x, y = xs[i], ys[j]
                tiles.append(box(x, y, x + tile_w, y + tile_h))
        pt_counts = 4
    return gpd.GeoDataFrame({"geometry": tiles, "pt_count": pt_counts})
