from __future__ import annotations

import geopandas as gpd
import numpy as np
from shapely.ops import unary_union
from shapely.strtree import STRtree


def iou(a, b):
    inter = a.intersection(b).area
    union = a.union(b).area
    return inter / union if union != 0 else 0


def preprocess_gdf(gdf: gpd.GeoDataFrame, buffer_px: float = 0) -> gpd.GeoDataFrame:
    """Preprocess the :term:`polygons <polygon>` by applying a buffer and filtering invalid geometries."""
    new_gdf = gdf.copy()
    if buffer_px != 0:
        new_gdf["geometry"] = gdf["geometry"].buffer(buffer_px)
    else:
        invalid = ~new_gdf["geometry"].is_valid
        if invalid.any():
            geometry = new_gdf["geometry"].copy()
            geometry.loc[invalid] = geometry.loc[invalid].buffer(0)
            new_gdf["geometry"] = geometry
    # Filter out invalid and empty geometries efficiently
    return new_gdf[new_gdf["geometry"].is_valid & ~new_gdf["geometry"].is_empty]


def nms(
    gdf: gpd.GeoDataFrame,
    prob_col: str,
    iou_threshold: float = 0.2,
    buffer_px: float = 0,
) -> gpd.GeoDataFrame:
    """
    Performs :term:`non-maximum suppression` (:term:`NMS`) on a :term:`GeoDataFrame` containing :term:`polygon` geometries.

    This function is primarily used to reduce overlapping :term:`polygons <polygon>` by selecting only the
    most probable geometric shapes according to the specified probability column. :term:`NMS`
    process is guided by the :term:`intersection over union` (:term:`IoU`) threshold, buffering distance,
    and probability values associated with :term:`polygons <polygon>`.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        A :term:`GeoDataFrame` containing geometric data for processing.
    prob_col : str
        Column name containing the probability values that determine
        the importance of each :term:`polygon`.
    iou_threshold : float
        A threshold value that defines the minimum
        :term:`IoU` for :term:`polygons <polygon>` to be considered for suppression. Default is 0.2.
    buffer_px : float
        Buffer distance for :term:`polygons <polygon>` before performing
        :term:`non-maximum suppression`. Default is 0.

    Returns
    -------
    gpd.GeoDataFrame
        A new :term:`GeoDataFrame` containing only the selected :term:`polygons <polygon>`
        after applying :term:`non-maximum suppression`.
    """

    # Work on a clean positional index so STRtree positions map back to ``gdf``
    # rows correctly. ``preprocess_gdf`` may DROP invalid/empty geometries, so the
    # tree is built over a subset; ``valid_pos`` records each kept polygon's
    # original row position in ``gdf``. Indexing ``gdf`` by tree positions directly
    # (the previous behaviour) silently selected the wrong rows whenever any row
    # was dropped.
    gdf = gdf.reset_index(drop=True)
    pp_gdf = preprocess_gdf(gdf, buffer_px=buffer_px)
    valid_pos = pp_gdf.index.to_numpy()  # original gdf positions of kept polygons
    polygons = pp_gdf["geometry"].tolist()
    if len(polygons) == 0:
        return gdf.iloc[[]]

    probs = gdf[prob_col].to_numpy()
    tree = STRtree(polygons)
    merged, suppressed = set(), set()

    # Iterate by tree position (an int), NOT by geometry object: a set membership
    # test against geometry objects never matched, so suppression was a no-op.
    for k in range(len(polygons)):
        if k in suppressed:
            continue
        geom = polygons[k]
        while True:
            cand = np.array(
                [
                    int(g)
                    for g in tree.query(geom, predicate="intersects")
                    if int(g) not in suppressed
                ],
                dtype=int,
            )
            if iou_threshold > 0 and cand.size:
                ious = np.array([iou(geom, polygons[j]) for j in cand])
                cand = cand[ious > iou_threshold]
            if cand.size == 0:
                break
            elif cand.size == 1:
                merged.add(int(cand[0]))
                break
            else:
                # Keep the highest-probability polygon in the group, suppress rest
                best = int(cand[np.argmax(probs[valid_pos[cand]])])
                merged.add(best)
                suppressed.update(int(c) for c in cand if c != best)
                geom = polygons[best]

    keep = valid_pos[np.array(sorted(merged), dtype=int)]
    return gdf.iloc[keep]


def merge_connected_polygons(
    gdf: gpd.GeoDataFrame,
    prob_col: str = None,
    buffer_px: float = 0,
):
    """
    Merge :term:`polygons <polygon>` in a :term:`GeoDataFrame` while optionally considering a probability column and applying a buffer.

    Parameters
    ----------
    gdf : :class:`GeoDataFrame <geopandas.GeoDataFrame>`
        :term:`GeoDataFrame` containing the input geometries to merge.
    prob_col : str, optional
        Name of the column containing the probability values. Default is None.
    buffer_px : float, optional
        Buffer distance applied during preprocessing of the geometry. Default is 0.

    Returns
    -------
    :class:`GeoDataFrame <geopandas.GeoDataFrame>`
        :term:`GeoDataFrame` containing the merged geometries, along with probability data if the prob_col parameter is set.
    """
    pp_gdf = preprocess_gdf(gdf, buffer_px=buffer_px)
    has_prob = prob_col in gdf.columns if prob_col else False

    polygons = pp_gdf["geometry"].tolist()
    merged, probs = [], []

    tree = STRtree(polygons)
    visited = set()

    for geom in polygons:
        if geom in visited:
            continue

        groups_ix = set(
            [g for g in tree.query(geom, predicate="intersects") if g not in visited]
        )
        if len(groups_ix) == 0:
            continue
        else:
            # continue finding other polygons that intersect with the group
            # until the group size is stable
            current_group_size = len(groups_ix)
            while True:
                new_groups_ix = set()
                for ix in groups_ix:
                    c_groups_ix = tree.query(polygons[ix], predicate="intersects")
                    # Intersects but not touches
                    c_groups_ix = [g for g in c_groups_ix if g not in visited]
                    new_groups_ix.update(c_groups_ix)
                groups_ix.update(new_groups_ix)
                if len(groups_ix) == current_group_size:
                    break
                current_group_size = len(groups_ix)

            # Sort the group index
            groups_ix = np.sort(list(groups_ix))
            visited.update(groups_ix)

            # Merge the group
            if len(groups_ix) == 1:
                ix = groups_ix[0]
                m_geom = polygons[ix]
                merged.append(m_geom)
                if has_prob:
                    prob = gdf.iloc[ix][prob_col]
                    probs.append(prob)
            else:
                m_geoms = [polygons[g] for g in groups_ix]
                m_geom = unary_union(m_geoms).buffer(0)
                if m_geom.is_valid & (m_geom.is_empty is False):
                    merged.append(m_geom)
                    if has_prob:
                        gs_gdf = gdf.iloc[groups_ix]
                        prob = np.average(
                            gs_gdf[prob_col], weights=gs_gdf["geometry"].area
                        )
                        probs.append(prob)

    data = {"geometry": merged}
    if has_prob:
        data[prob_col] = probs
    merged_gdf = gpd.GeoDataFrame(data, crs=None)
    # Offset the geometry if buffer_px is set
    if buffer_px > 0:
        merged_gdf["geometry"] = merged_gdf["geometry"].buffer(-buffer_px).buffer(0)
    return merged_gdf
