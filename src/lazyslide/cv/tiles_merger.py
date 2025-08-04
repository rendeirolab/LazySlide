from __future__ import annotations

import geopandas as gpd
import numpy as np
from shapely.ops import unary_union
from shapely.strtree import STRtree

# There are two ways to merge polygons:
# 1. For semantic segmentation: Merge overlapping polygons using a spatial index (STRtree)
# 2. For cell segmentation: Only preserve the largest polygon in each group of overlapping polygons.


def iou(a, b):
    inter = a.intersection(b).area
    union = a.union(b).area
    return inter / union if union != 0 else 0


def preprocess_gdf(gdf: gpd.GeoDataFrame, buffer_px: float = 0) -> gpd.GeoDataFrame:
    """Preprocess the polygons by applying a buffer and filtering invalid geometries."""
    new_gdf = gdf.copy()
    new_gdf["geometry"] = gdf["geometry"].buffer(buffer_px)
    # Filter out invalid and empty geometries efficiently
    return new_gdf[new_gdf["geometry"].is_valid & ~new_gdf["geometry"].is_empty]


def nms(
    gdf: gpd.GeoDataFrame,
    prob_col: str,
    iou_threshold: float = 0.2,
    buffer_px: float = 0,
) -> gpd.GeoDataFrame:
    """
    Performs non-maximum suppression (NMS) on a GeoDataFrame containing polygon geometries.

    This function is primarily used to reduce overlapping polygons by selecting only the
    most probable geometric shapes according to the specified probability column. NMS
    process is guided by the intersection over union (IoU) threshold, buffering distance,
    and probability values associated with polygons.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        A GeoDataFrame containing geometric data for processing.
    prob_col : str
        Column name containing the probability values that determine
        the importance of each polygon.
    iou_threshold : float
        A threshold value that defines the minimum
        IoU for polygons to be considered for suppression. Default is 0.2.
    buffer_px : float
        Buffer distance for polygons before performing
        non-maximum suppression. Default is 0.

    Returns
    -------
    gpd.GeoDataFrame
        A new GeoDataFrame containing only the selected polygons
        after applying non-maximum suppression.
    """

    pp_gdf = preprocess_gdf(gdf, buffer_px=buffer_px)
    polygons = pp_gdf["geometry"].tolist()

    tree = STRtree(polygons)
    merged, suppressed = set(), set()

    for geom in polygons:
        if geom in suppressed:
            continue
        while True:
            groups_ix = [
                g
                for g in tree.query(geom, predicate="intersects")
                if g not in suppressed
            ]
            groups_ix = np.array(groups_ix)
            if iou_threshold > 0:
                ious = np.array([iou(geom, polygons[i]) for i in groups_ix])
                groups_ix = groups_ix[ious > iou_threshold]
            n_groups = len(groups_ix)
            if n_groups == 0:
                break
            elif n_groups == 1:
                merged.add(groups_ix[0])
                break
            else:
                # Find the highest probability polygon in the group
                probs = [pp_gdf.loc[ix, prob_col] for ix in groups_ix]
                largest_ix = groups_ix[np.argmax(probs)]
                merged.add(largest_ix)
                # Remove largest ix from the group ix
                groups_ix = list(groups_ix)
                groups_ix.remove(largest_ix)
                suppressed.update(groups_ix)
                geom = polygons[largest_ix]

    return gdf.iloc[list(merged)]


def merge_connected_polygons(
    gdf: gpd.GeoDataFrame,
    prob_col: str = None,
    buffer_px: float = 0,
):
    """
    Merge polygons in a GeoDataFrame while optionally considering a probability column and applying a buffer.

    Parameters
    ----------
    gdf : :class:`GeoDataFrame <geopandas.GeoDataFrame>`
        GeoDataFrame containing the input geometries to merge.
    prob_col : str, optional
        Name of the column containing the probability values. Default is None.
    buffer_px : float, optional
        Buffer distance applied during preprocessing of the geometry. Default is 0.

    Returns
    -------
    :class:`GeoDataFrame <geopandas.GeoDataFrame>`
        GeoDataFrame containing the merged geometries, along with probability data if the prob_col parameter is set.
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
