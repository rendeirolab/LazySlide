from __future__ import annotations

from itertools import combinations

import geopandas as gpd
import numpy as np
from shapely.ops import unary_union
from shapely.strtree import STRtree


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
    drop_overlap : float, default: 0.9
        The ratio to drop the overlapping polygons.

    """

    def __init__(
        self,
        gdf: gpd.GeoDataFrame,
        class_col: str = None,
        prob_col: str = None,
        buffer_px: float = 0,
        drop_overlap: float = 0.9,
    ):
        self.gdf = gdf
        self.class_col = class_col
        self.prob_col = prob_col
        self.buffer_px = buffer_px
        self.drop_overlap = drop_overlap

        self._has_class = class_col in gdf.columns if class_col else False
        self._has_prob = prob_col in gdf.columns if prob_col else False
        self._preprocessed_polygons = self._preprocess_polys()
        self._merged_polygons = None

    def _preprocess_polys(self):
        """Preprocess the polygons."""
        new_gdf = self.gdf.copy()
        if self.buffer_px > 0:
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
            merged_geoms = []  # (polygon, row_ix, groups_ix)

            if len(groups_ix) == 1:
                ix = groups_ix[0]
                m_geoms = polygons[ix]
                merged_geoms.append((m_geoms, ix, groups_ix))
            else:
                m_geoms = [polygons[g] for g in groups_ix]
                if self._has_class:
                    ref_df = gpd.GeoDataFrame(
                        {
                            "names": [gdf[self.class_col].values[g] for g in groups_ix],
                            "index": groups_ix,
                            "geometry": m_geoms,
                        }
                    )

                    # {class_name: polygon}
                    named_polys = (
                        ref_df[["names", "geometry"]]
                        .groupby("names")
                        .apply(unary_union)
                        .to_dict()
                    )

                    if self.drop_overlap > 0:
                        # If the two classes instances are more than 90% overlapping
                        # The smaller one is removed
                        while len(named_polys) > 1:
                            names = list(named_polys.keys())
                            combs = combinations(names, 2)
                            for n1, n2 in combs:
                                if n1 in named_polys and n2 in named_polys:
                                    p1, p2 = named_polys[n1], named_polys[n2]
                                    if p1.intersection(p2).is_empty:
                                        continue
                                    area, drop = (
                                        (p1.area, n1)
                                        if p1.area < p2.area
                                        else (p2.area, n2)
                                    )
                                    union = p1.union(p2).area
                                    overlap_ratio = union / area
                                    if overlap_ratio > self.drop_overlap:
                                        del named_polys[drop]
                            break
                    for n, p in named_polys.items():
                        gs = ref_df[ref_df["names"] == n]["index"].tolist()
                        merged_geoms.append((p, gs[0], gs))
                else:
                    m_geoms = unary_union(m_geoms)
                    merged_geoms.append((m_geoms, groups_ix[0], groups_ix))
            # Postprocess the merged polygon
            for m_geom, ix, gs_ix in merged_geoms:
                if self.buffer_px > 0:
                    m_geom = m_geom.buffer(-self.buffer_px).buffer(0)
                if m_geom.is_valid & (m_geom.is_empty is False):
                    m_data = gdf.iloc[ix].copy()
                    m_data["geometry"] = m_geom
                    if self._has_prob:
                        gs_gdf = gdf.iloc[gs_ix]
                        m_data[self.prob_col] = np.average(
                            gs_gdf[self.prob_col], weights=gs_gdf["geometry"].area
                        )
                    merged.append(m_data)
            for g in groups_ix:
                visited.add(g)
        return gpd.GeoDataFrame(merged)

    def merge(self):
        """Launch the merging process."""
        self._merged_polygons = self._tree_merge(self._preprocessed_polygons)

    @property
    def merged_polygons(self):
        return self._merged_polygons


def merge_polygons(
    gdf: gpd.GeoDataFrame,
    class_col: str = None,
    prob_col: str = None,
    buffer_px: float = 0,
    drop_overlap: float = 0.9,
):
    merger = PolygonMerger(gdf, class_col, prob_col, buffer_px, drop_overlap)
    merger.merge()
    return merger.merged_polygons


merge_polygons.__doc__ = PolygonMerger.__doc__
