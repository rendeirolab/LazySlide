from __future__ import annotations

import warnings
from typing import Literal, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from spatialdata.models import ShapesModel
from wsidata import TileSpec, WSIData

from lazyslide._const import Key
from lazyslide._utils import find_stack_level


def tile_tissues(
    wsi: WSIData,
    tile_px: int | Tuple[int, int],
    *,
    stride_px: int | Tuple[int, int] | None = None,
    overlap: float = None,
    edge: bool = False,
    mpp: float = None,
    slide_mpp: float = None,
    ops_level: int = None,
    background_filter: bool = True,
    background_fraction: float = 0.3,
    background_filter_mode: Literal["approx", "exact"] | None = None,
    tissue_key: str | None = Key.tissue,
    key_added: str | None = Key.tiles,
    return_tiles: bool = False,
):
    """
    Generate :term:`tiles <tile>` within the tissue :term:`contours` in the :term:`WSI`.

    If there is no tissue contours, the tiles will generate for the whole image.

    Parameters
    ----------
    wsi : :class:`WSIData <wsidata.WSIData>`
        The :term:`WSIData` object to work on.
    tile_px : int or tuple of (int, int)
        The size of the tile, if tuple, (W, H).
    stride_px : int or tuple of (int, int), default: None
        The stride of the tile, if tuple, (W, H).
        If None, use the tile size.
    overlap : float, default: None
        The overlap of the tiles, exclusive with stride_px.
        If in (0, 1), it's the overlap ratio.
        If > 1, it's the overlap in pixels.
    edge : bool, default: False
        Whether to include the edge tiles.
    mpp : float, default: None
        The requested :term:`mpp` of the tiles, if None, use the slide mpp.
    slide_mpp : float, default: None
        This value will override the slide mpp.
    ops_level : int, default: None
        Which level to use for the actual tile image retrival.
    background_filter : bool, default: True
        Whether to filter the tiles based on the background fraction.
    background_fraction : float, default: 0.3
        Only used if `background_filter` is True.
        The fraction of background in the tile, if more than this, discard the tile.
    background_filter_mode : {'approx', 'exact'}, optional
        .. deprecated::
            No longer has any effect. Background filtering is now always exact
            and vectorized: tiles fully inside the tissue are kept directly and
            border tiles are filtered by their exact tissue-coverage fraction.
    tissue_key : str, default: 'tissues'
        The key of the tissue contours.
    key_added : str, default: 'tiles'
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

    if background_filter_mode is not None:
        warnings.warn(
            "`background_filter_mode` is deprecated and no longer has any effect. "
            "Background filtering is now always exact and vectorized.",
            DeprecationWarning,
            stacklevel=find_stack_level(),
        )

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
        if not cnt.is_valid:
            cnt = cnt.buffer(0)
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
            # Tiles the tissue fully contains are 100% covered -> keep them
            # directly. The rest (border tiles) get an exact, vectorized
            # coverage check. `contains` handles concave tissue and holes
            # correctly, so the filtering is exact everywhere.
            geoms = tiles.geometry.to_numpy()
            shapely.prepare(cnt)
            tree = shapely.STRtree(geoms)
            keep = np.zeros(len(geoms), dtype=bool)
            keep[tree.query(cnt, predicate="contains")] = True
            border = np.flatnonzero(~keep)
            if border.size:
                bgeoms = geoms[border]
                cov = shapely.area(shapely.intersection(bgeoms, cnt)) / shapely.area(
                    bgeoms
                )
                keep[border] = cov >= (1 - background_fraction)
            overlap_tiles = tiles.iloc[keep]

        else:
            # If no background filter, keep all intersecting tiles
            overlap_tiles = tiles

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


def _starts(origin, length, tile, stride, edge):
    """Tile start coordinates along one axis covering ``[origin, origin+length]``.

    A tile spans ``[start, start + tile]`` and fits inside the bounding box when
    ``start <= origin + (length - tile)``. With ``edge=True`` a single extra tile
    is appended that overruns the box to cover the trailing remainder.
    """
    if length < tile:
        # The box is smaller than one tile: emit a single tile at the origin.
        n_fit = 1
        last_fit_start = origin
    else:
        n_fit = (length - tile) // stride + 1
        last_fit_start = origin + (n_fit - 1) * stride
    starts = origin + np.arange(n_fit, dtype=np.int64) * stride
    if edge and length >= tile and (last_fit_start + tile) < (origin + length):
        starts = np.append(starts, last_fit_start + stride)
    return starts


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
    # A vectorized implementation using shapely 2.x array ufuncs.
    x, y, w, h = int(x), int(y), int(w), int(h)

    if stride_w is None:
        stride_w = tile_w
    if stride_h is None:
        stride_h = tile_h

    xs = _starts(x, w, tile_w, stride_w, edge)
    ys = _starts(y, h, tile_h, stride_h, edge)

    # Build all candidate tile boxes at once
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    x0 = gx.ravel()
    y0 = gy.ravel()
    x1 = x0 + tile_w
    y1 = y0 + tile_h
    boxes = shapely.box(x0, y0, x1, y1)

    if mask is None:
        return gpd.GeoDataFrame(geometry=boxes)

    # Keep every tile that truly intersects the tissue. An STRtree
    # intersects-query (instead of corner sampling) avoids dropping tiles whose
    # interior overlaps tissue while none of their corners are inside it
    # (thin strips, tissue islands smaller than a tile).
    shapely.prepare(mask)
    keep = np.zeros(boxes.shape[0], dtype=bool)
    keep[shapely.STRtree(boxes).query(mask, predicate="intersects")] = True
    return gpd.GeoDataFrame(geometry=boxes[keep])
