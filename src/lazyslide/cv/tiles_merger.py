from __future__ import annotations

from typing import Generator, List

import geopandas as gpd
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree

# There are two ways to merge polygons:
# 1. For semantic segmentation: Merge overlapping polygons using a spatial index (STRtree)
# 2. For cell segmentation: Only preserve the largest polygon in each group of overlapping polygons.


def polygon_groups(polygons: List[Polygon]) -> Generator[NDArray[np.integer]]:
    """A generator that yields indexes of polygon that are intersected."""
    tree = STRtree(polygons)
    visited = set()

    for geom in polygons:
        if geom in visited:
            continue

        groups_ix = tree.query(geom, predicate="intersects")
        groups_ix = set([g for g in groups_ix if g not in visited])
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
                    c_groups_ix = [g for g in c_groups_ix if g not in visited]
                    new_groups_ix.update(c_groups_ix)
                groups_ix.update(new_groups_ix)
                if len(groups_ix) == current_group_size:
                    break
                current_group_size = len(groups_ix)

        # Sort the group index
        groups_ix = np.sort(list(groups_ix))
        visited.update(groups_ix)
        yield groups_ix


def preprocess_gdf(gdf: gpd.GeoDataFrame, buffer_px: float = 0) -> gpd.GeoDataFrame:
    """Preprocess the polygons by applying a buffer and filtering invalid geometries."""
    new_gdf = gdf.copy()
    new_gdf["geometry"] = gdf["geometry"].buffer(buffer_px)
    # Filter out invalid and empty geometries efficiently
    return new_gdf[new_gdf["geometry"].is_valid & ~new_gdf["geometry"].is_empty]


class PolygonMerger:
    """
    Merge polygons from different tiles.

    If the polygons are overlapping/touching, the overlapping regions are merged.

    If probabilities exist, the probabilities are averaged weighted by the area of the polygons.

    Parameters
    ----------
    gdf : `GeoDataFrame <geopandas.GeoDataFrame>`
        The GeoDataFrame containing the polygons.
    class_col : str, default: None
        The column that specify the names of the polygons.
    prob_col : str, default: None
        The column that specify the probabilities of the polygons.
    buffer_px : float, default: 0
        The buffer size for the polygons to test the intersection.

    """

    def __init__(
        self,
        gdf: gpd.GeoDataFrame,
        class_col: str = None,
        prob_col: str = None,
        buffer_px: float = 0,
    ):
        self.gdf = gdf
        self.class_col = class_col
        self.prob_col = prob_col
        self.buffer_px = buffer_px

        self._has_class = class_col in gdf.columns if class_col else False
        self._has_prob = prob_col in gdf.columns if prob_col else False
        self._preprocessed_polygons = self._preprocess_polys()
        self._merged_polygons = None

    def _preprocess_polys(self):
        """Preprocess the polygons."""
        new_gdf = self.gdf.copy()
        new_gdf["geometry"] = self.gdf["geometry"].buffer(self.buffer_px)
        # Filter out invalid and empty geometries efficiently
        return new_gdf[new_gdf["geometry"].is_valid & ~new_gdf["geometry"].is_empty]

    def _merge_overlap(self, gdf: gpd.GeoDataFrame):
        """
        Merge the overlapping polygons recursively.

        This function has no assumptions about the class or probability
        """
        pass

    def _tree_merge(self, gdf: gpd.GeoDataFrame):
        polygons = gdf["geometry"].tolist()
        tree = STRtree(polygons)
        visited = set()
        merged = []

        for geom in polygons:
            if geom in visited:
                continue

            groups_ix = tree.query(geom, predicate="intersects")
            groups_ix = set([g for g in groups_ix if g not in visited])
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
                        c_groups_ix = [g for g in c_groups_ix if g not in visited]
                        new_groups_ix.update(c_groups_ix)
                    groups_ix.update(new_groups_ix)
                    if len(groups_ix) == current_group_size:
                        break
                    current_group_size = len(groups_ix)

            # Sort the group index
            groups_ix = np.sort(list(groups_ix))

            # Merge the group
            if len(groups_ix) == 1:
                ix = groups_ix[0]
                m_geom = polygons[ix]
                if self.buffer_px > 0:
                    m_geom = m_geom.buffer(-self.buffer_px).buffer(0)
                else:
                    m_geom = m_geom.buffer(0)
                record = {"geometry": m_geom}
                if self._has_prob:
                    prob = gdf.iloc[ix][self.prob_col]
                    record[self.prob_col] = prob
                merged.append(record)
            else:
                m_geoms = [polygons[g] for g in groups_ix]
                # if self._has_class:
                #     ref_df = gpd.GeoDataFrame(
                #         {
                #             "names": [gdf[self.class_col].values[g] for g in groups_ix],
                #             "index": groups_ix,
                #             "geometry": m_geoms,
                #         }
                #     )
                #
                #     # {class_name: polygon}
                #     named_polys = (
                #         ref_df[["names", "geometry"]]
                #         .groupby("names")
                #         .apply(unary_union)
                #         .to_dict()
                #     )
                #
                #     if self.drop_overlap > 0:
                #         # If the two classes instances are more than 90% overlapping
                #         # The smaller one is removed
                #         while len(named_polys) > 1:
                #             names = list(named_polys.keys())
                #             combs = combinations(names, 2)
                #             for n1, n2 in combs:
                #                 if n1 in named_polys and n2 in named_polys:
                #                     p1, p2 = named_polys[n1], named_polys[n2]
                #                     if p1.intersection(p2).is_empty:
                #                         continue
                #                     area, drop = (
                #                         (p1.area, n1)
                #                         if p1.area < p2.area
                #                         else (p2.area, n2)
                #                     )
                #                     union = p1.union(p2).area
                #                     overlap_ratio = union / area
                #                     if overlap_ratio > self.drop_overlap:
                #                         del named_polys[drop]
                #             break
                #     for n, p in named_polys.items():
                #         gs = ref_df[ref_df["names"] == n]["index"].tolist()
                #         merged_geoms.append((p, gs[0], gs))
                # else:
                m_geom = unary_union(m_geoms)
                if self.buffer_px > 0:
                    m_geom = m_geom.buffer(-self.buffer_px).buffer(0)
                else:
                    m_geom = m_geom.buffer(0)

                if m_geom.is_valid & (m_geom.is_empty is False):
                    record = {"geometry": m_geom}
                    if self._has_prob:
                        gs_gdf = gdf.iloc[groups_ix]
                        prob = np.average(
                            gs_gdf[self.prob_col], weights=gs_gdf["geometry"].area
                        )
                        record[self.prob_col] = prob
                    merged.append(record)
            visited.update(groups_ix)
        return gpd.GeoDataFrame(merged)

    def merge(self):
        """Launch the merging process."""
        results = []
        if self._has_class:
            for c, polys in self._preprocessed_polygons.groupby(self.class_col):
                merged_polys = self._tree_merge(polys)
                merged_polys[self.class_col] = c
                results.append(merged_polys)
        else:
            merged_polys = self._tree_merge(self._preprocessed_polygons)
            results.append(merged_polys)

        self._merged_polygons = gpd.GeoDataFrame(
            pd.concat(results, ignore_index=True)
        ).reset_index(drop=True)

    @property
    def merged_polygons(self):
        return self._merged_polygons


def preserve_largest_polygon(
    gdf: gpd.GeoDataFrame, buffer_px: float = 0
) -> gpd.GeoDataFrame:
    """
    Preserve the largest polygon in each group of overlapping polygons.

    Parameters
    ----------
    gdf : :class:`GeoDataFrame <geopandas.GeoDataFrame>`
        The GeoDataFrame containing the polygons.
    buffer_px : float, default: 0
        The buffer size for the polygons to test the intersection.

    Returns
    -------
    :class:`GeoDataFrame <geopandas.GeoDataFrame>`
        The GeoDataFrame with the largest polygons preserved.
    """

    pp_gdf = preprocess_gdf(gdf, buffer_px=buffer_px)
    polygons = pp_gdf["geometry"].tolist()

    merged = []
    for groups_ix in polygon_groups(polygons):
        if len(groups_ix) == 1:
            merged.append(gdf.iloc[groups_ix[0]])
        else:
            # Find the largest polygon in the group
            areas = [gdf.iloc[ix].geometry.area for ix in groups_ix]
            largest_ix = groups_ix[np.argmax(areas)]
            merged.append(gdf.iloc[largest_ix])

    return gpd.GeoDataFrame(merged)


def merge_polygons(
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

    merged = []
    probs = []
    for groups_ix in polygon_groups(polygons):
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
                    prob = np.average(gs_gdf[prob_col], weights=gs_gdf["geometry"].area)
                    probs.append(prob)
    data = {"geometry": merged}
    if has_prob:
        data[prob_col] = probs
    merged_gdf = gpd.GeoDataFrame(data, crs=None)
    # Offset the geometry if buffer_px is set
    if buffer_px > 0:
        merged_gdf["geometry"] = merged_gdf["geometry"].buffer(-buffer_px).buffer(0)
    return merged_gdf
