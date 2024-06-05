from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Dict, Sequence

import numpy as np
import pandas as pd
from fsspec import open
from fsspec.core import OpenFile
from pydantic import BaseModel

from .reader import get_reader

import lazy_loader as lazy

ad = lazy.load("anndata")


class TileSpec(BaseModel):
    level: int = 0
    downsample: float = 1
    mpp: Optional[float] = None
    height: int
    width: int
    ops_height: int
    ops_width: int
    tissue_name: str


class SlideData:
    def __init__(
        self,
        file: str | Path,
    ):
        from spatialdata import SpatialData
        from spatialdata import read_zarr

        self.file = str(file)
        open_file: OpenFile = open(self.file)
        fs = open_file.fs

        if fs.exists(self.file):
            self.sdata = read_zarr(self.file)
        else:
            self.sdata = SpatialData()

        open_file.close()

    def add_image(self, image: np.ndarray, level: int, dims=("c", "y", "x")):
        from spatialdata.models import Image2DModel

        image_adata = Image2DModel.parse(image, dims=dims)
        self.sdata.images[f"level_{level}"] = image_adata

    def add_attributes(self, attributes: dict, name="attributes"):
        if name in self.sdata.tables:
            self.sdata.tables[name].uns.update(attributes)
        else:
            attrs = ad.AnnData(uns=attributes)
            self.sdata.tables[name] = attrs

    def _add_shape(self, shapes, name, data=None):
        import geopandas as gpd
        from shapely.geometry import Polygon
        from spatialdata.models import ShapesModel

        gdf = gpd.GeoDataFrame(geometry=[Polygon(c) for c in shapes], data=data)
        cs = ShapesModel.parse(gdf)
        self.sdata.shapes[name] = cs

    def add_contours(self, contours: list[np.ndarray], data=None, name="contours"):
        self._add_shape(contours, name, data)

    def add_holes(self, holes: list[np.ndarray], data=None, name="holes"):
        self._add_shape(holes, name, data)

    def add_tiles(self, xy: np.ndarray, name: str, spec: TileSpec | dict, data=None):
        """
        Add tiles to the slide data.

        Parameters
        ----------
        xy : np.ndarray
            An array of tile coordinates.
        name: str
            The name of the tile.
        spec : TileSpec | dict
            The tile specification.

        """
        from spatialdata.models import PointsModel

        if data is not None:
            annotation = pd.DataFrame(data)
        else:
            annotation = None
        tiles = PointsModel.parse(xy.astype(int), annotation=annotation)
        self.sdata.points[name] = tiles
        # Parse spec
        if isinstance(spec, dict):
            spec = TileSpec(**spec)
        spec = spec.model_dump()
        ref_adata = ad.AnnData(uns={"tile_spec": spec})
        self.sdata.tables[f"{name}_spec"] = ref_adata

    def add_tile_annotations(self, annotations: Dict[str, Sequence], name: str):
        import dask.array as da

        if name not in self.sdata.points:
            raise ValueError(f"Tile {name} not found.")
        tiles = self.sdata.points[name]
        for key, value in annotations.items():
            tiles[key] = da.from_array(value)

    def add_features(
        self, features: np.ndarray | pd.DataFrame, tile_name: str, feature_name: str
    ):
        # Check if the tile exists
        if tile_name not in self.sdata.points:
            raise ValueError(f"Tile {tile_name} not found.")
        # Check if the features are correct
        # if len(features) != self.sdata.tables[tile_name].shape[0]:
        #     raise ValueError(f"Features length does not match the tile {tile_name}.")

        feature_adata = ad.AnnData(X=features)
        self.sdata.tables[f"{tile_name}/{feature_name}"] = feature_adata

    def add_metadata(self, metadata: dict):
        if "metadata" in self.sdata.tables:
            self.sdata.tables["metadata"].uns["metadata"].update(metadata)
        self.sdata.tables["metadata"] = ad.AnnData(uns={"metadata": metadata})

    def to_tile_anndata(self, tile_name: str, feature_name: str = None):
        if feature_name is not None:
            feature_table = self.sdata.tables[f"{tile_name}/{feature_name}"]
            X = feature_table.X
            var = feature_table.var
        else:
            X = None
            var = None

        tile_table = self.sdata.tables[f"{tile_name}_table"]
        spec_table = self.sdata.tables[f"{tile_name}_spec"]
        obs = tile_table.obs
        return ad.AnnData(
            X=X, obs=obs, var=var, uns=spec_table.uns, obsm=spec_table.obsm
        )

    def write(self, file=None, overwrite=True, **kws):
        if file is None:
            file = self.file
        self.sdata.write(file, overwrite=overwrite, **kws)

    def is_backed(self):
        return self.sdata.is_backed()


class WSI(SlideData):
    def __init__(self, slide: Any, backed_file=None, reader="auto", **kwargs):
        # TODO: Use fsspec to download remote slide to temporary file
        self.slide = str(slide)
        open(self.slide)
        reader_cls = get_reader(reader)
        self.reader = reader_cls(self.slide, **kwargs)
        if backed_file is None:
            self.backed_file = Path(slide).with_suffix(".zarr")
        super().__init__(self.backed_file)

    @property
    def metadata(self):
        return self.reader.metadata

    def get_tile_spec(self, key) -> TileSpec:
        return TileSpec(**self.sdata.tables[f"{key}_spec"].uns["tile_spec"])

    def get_region(self, x, y, width, height, level=0, **kwargs):
        return self.reader.get_region(x, y, width, height, level=level, **kwargs)
