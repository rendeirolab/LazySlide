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
    names : str, default: 'names'
        The column that specify the names of the polygons.
    probs : str, default: 'probs'
        The column that specify the probabilities of the polygons.
    buffer_px : float, default: 0
        The buffer size for the polygons to test the intersection.
    task : {'multilabel', 'multiclass'}, default: 'multilabel'
        The task type of the segmentation.
        If 'multiclass', the same pixel can be assigned to multiple polygons.
        There will be overlapping polygons.
    drop_overlap : float, default: 0.9
        The ratio to drop the overlapping polygons.

    """

    def __init__(
        self,
        gdf: gpd.GeoDataFrame,
        name: str = "name",
        prob: str = "prob",
        buffer_px: float = 0,
        task: str = "multilabel",
        drop_overlap: float = 0.9,
    ):
        self.gdf = gdf
        self.name = name
        self.prob = prob
        self.buffer_px = buffer_px
        self.task = task
        self.drop_overlap = drop_overlap

        self._has_name = name in gdf.columns
        self._has_prob = prob in gdf.columns
        self._preprocessed_polygons = self._preprocess_polys()
        self._merged_polygons = None

    def _preprocess_polys(self):
        """Preprocess the polygons."""
        new_gdf = self.gdf.copy()
        if self.buffer_px > 0:
            new_gdf["geometry"] = self.gdf["geometry"].buffer(self.buffer_px)
        # Filter out invalid and empty geometries efficiently
        return new_gdf[new_gdf["geometry"].is_valid & ~new_gdf["geometry"].is_empty]

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
                if self._has_name and self.task == "multilabel":
                    ref_df = gpd.GeoDataFrame(
                        {
                            "names": [gdf[self.name].values[g] for g in groups_ix],
                            "index": groups_ix,
                            "geometry": m_geoms,
                        }
                    )

                    named_polys = (
                        ref_df[["names", "geometry"]]
                        .groupby("names")
                        .apply(unary_union)
                        .to_dict()
                    )
                    # If the merged polygons are overlapping
                    # with more than 90% of the area
                    # The smaller polygons are removed
                    while len(named_polys) > 1:
                        names = list(named_polys.keys())
                        combs = combinations(names, 2)
                        for n1, n2 in combs:
                            p1, p2 = named_polys[n1], named_polys[n2]
                            area, drop = (
                                (p1.area, n1) if p1.area < p2.area else (p2.area, n2)
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
                        m_data[self.prob] = np.average(
                            gs_gdf[self.prob], weights=gs_gdf["geometry"].area
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
    names: str = "names",
    probs: str = "probs",
    buffer_px: float = 0,
    task: str = "multilabel",
    drop_overlap: float = 0.9,
):
    merger = PolygonMerger(gdf, names, probs, buffer_px, task, drop_overlap)
    merger.merge()
    return merger.merged_polygons


merge_polygons.__doc__ = PolygonMerger.__doc__
