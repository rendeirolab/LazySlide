from typing import Sequence

from shapely.geometry.polygon import Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree


class PolygonMerger:
    """
    Merge polygons from different tiles.

    If the polygons are overlapping/touching, the overlapping regions are merged.

    """

    def __init__(
        self,
        polygons: Sequence[Polygon],
        names: Sequence[str] = None,
        buffer_px: float = 0,
    ):
        self.polygons = polygons
        self.names = names
        self.buffer_px = buffer_px
        self._preprocessed_polygons = self._preprocess_polys(polygons)

        self._grouped_polys = {}
        self._trees = {}
        if names is None:
            # Build a single tree
            self._trees["default"] = STRtree(self._preprocessed_polygons)
            self._grouped_polys["default"] = self._preprocessed_polygons
        else:
            # Build trees for each name
            grouped_polys = {}
            for name, poly in zip(names, self._preprocessed_polygons):
                if name not in grouped_polys:
                    grouped_polys[name] = []
                grouped_polys[name].append(poly)
            for name, polys in grouped_polys.items():
                self._trees[name] = STRtree(polys)
            self._grouped_polys = grouped_polys

        self._merged_polygons = {}

    def _preprocess_polys(self, polys):
        """Preprocess the polygons."""
        processed = []
        for p in polys:
            if self.buffer_px > 0:
                p = p.buffer(self.buffer_px)
            if p.is_valid and not p.is_empty:
                processed.append(p)
        return processed

    def _postprocess_polys(self, polys):
        """Postprocess the polygons."""
        if self.buffer_px > 0:
            ppolys = []
            for poly in polys:
                p = poly.buffer(-self.buffer_px).buffer(0)
                if p.is_valid & (p.is_empty is False):
                    ppolys.append(p)
            return ppolys
        else:
            return polys

    def _tree_merge(self, tree, geoms):
        visited = set()
        merged = []

        for geom in geoms:
            if geom in visited:
                continue

            groups = tree.query(geom, predicate="intersects")
            groups = [g for g in groups if g not in visited]

            # Merge the group
            if len(groups) == 1:
                m_geoms = geoms[groups[0]]
            else:
                m_geoms = [geoms[g] for g in groups]
                m_geoms = unary_union(m_geoms)

            # Postprocess the merged polygon
            if self.buffer_px > 0:
                m_geoms = m_geoms.buffer(-self.buffer_px).buffer(0)
            if m_geoms.is_valid & (m_geoms.is_empty is False):
                merged.append(m_geoms)
            for g in groups:
                visited.add(g)
        return merged

    def merge(self):
        """Launch the merging process."""
        for name, polys in self._grouped_polys.items():
            self._merged_polygons[name] = self._tree_merge(self._trees[name], polys)

    @property
    def merged_polygons(self):
        if len(self._merged_polygons) == 0:
            return None
        elif len(self._merged_polygons) == 1:
            if "default" in self._merged_polygons:
                return self._merged_polygons["default"]
            else:
                return self._merged_polygons
        else:
            return self._merged_polygons
