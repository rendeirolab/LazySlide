from __future__ import annotations

import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional, Dict, Sequence, Iterable

import lazy_loader as lazy
import numpy as np
import pandas as pd
from fsspec import open
from fsspec.core import OpenFile, url_to_fs
from fsspec.implementations.cached import WholeFileCacheFileSystem
from fsspec.implementations.local import LocalFileSystem

from .reader import get_reader

ad = lazy.load("anndata")


@dataclass
class TileSpec:
    height: int
    width: int
    raw_height: int
    raw_width: int
    tissue_name: str
    level: int = 0
    downsample: float = 1
    mpp: Optional[float] = None


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
            self._sdata = read_zarr(self.file)
        else:
            self._sdata = SpatialData()

        open_file.close()

    @property
    def sdata(self):
        return self._sdata

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

    def add_shapes(
        self,
        shapes: Iterable[np.ndarray],
        name: str,
        data: Dict[str, Sequence],
    ):
        import geopandas as gpd
        from shapely.geometry import Polygon
        from spatialdata.models import ShapesModel

        data = pd.DataFrame(data)
        gdf = gpd.GeoDataFrame(data=data, geometry=[Polygon(c) for c in shapes])
        cs = ShapesModel.parse(gdf)
        self.sdata.shapes[name] = cs

        # data["id"] = np.arange(data.shape[0])
        # data["shape_key"] = name
        # data["shape_key"] = data["shape_key"].astype("category")
        # data.index = data.index.astype(str)
        # shape_adata = ad.AnnData(obs=data)
        # shape_adata = TableModel.parse(
        #     shape_adata, region=[name], region_key="shape_key", instance_key="id"
        # )
        # self.sdata.tables[f"{name}_table"] = shape_adata

    def add_shapes_data(self, data: Dict[str, Sequence], name: str):
        # table_name = f"{name}_table"
        # if table_name not in self.sdata.tables:
        #     raise ValueError(f"Shape {table_name} not found.")
        # shapes = self.sdata.tables[table_name]
        shapes = self.sdata.shapes[name].compute()
        for key, value in data.items():
            shapes[key] = value

    def get_shape_table(self, name):
        # Check if the shape exists
        if name not in self.sdata.shapes:
            raise ValueError(f"Shape {name} not found.")
        shapes = self.sdata.shapes[name]
        tables = self.sdata.tables[f"{name}_table"].obs.reset_index(drop=True)
        return pd.concat([shapes, tables], axis=1)

    def add_tiles(
        self, xy: np.ndarray, tissue_id: np.ndarray, name: str, spec: TileSpec | dict
    ):
        """
        Add tiles to the slide data.

        Parameters
        ----------
        xy : np.ndarray
            An array of tile coordinates.
        tissue_id : np.ndarray
            An array of tissue ids.
        name: str
            The name of the tile.
        spec : TileSpec | dict
            The tile specification.

        """
        from spatialdata.models import PointsModel, TableModel

        # Add to points field
        xy = xy.astype(int)
        tiles = PointsModel.parse(xy)
        self.sdata.points[name] = tiles

        # Add to tables field
        points = pd.DataFrame(xy, columns=["x", "y"])
        points["tissue_id"] = tissue_id
        points["id"] = np.arange(xy.shape[0])
        # To suppress warnings
        points["tile_key"] = name
        points["tile_key"] = points["tile_key"].astype("category")
        points.index = points.index.astype(str)

        # Parse spec
        if isinstance(spec, dict):
            spec = TileSpec(**spec)
        spec = asdict(spec)
        ref_adata = ad.AnnData(obs=points, uns={"tile_spec": spec})
        ref_adata = TableModel.parse(
            ref_adata,
            region=[name],
            region_key="tile_key",
            instance_key="id",
        )
        self.sdata.tables[f"{name}_table"] = ref_adata

    def add_tiles_data(self, data: Dict[str, Sequence], name: str):
        table_name = f"{name}_table"
        if table_name not in self.sdata.tables:
            raise ValueError(f"Tile {table_name} not found.")
        tiles = self.sdata.tables[table_name]
        for key, value in data.items():
            tiles.obs[key] = value

    def subset_tiles(self, name, new_name, indices, overwrite=False):
        if name not in self.sdata.points:
            raise ValueError(f"Tile {name} not found.")
        if new_name in self.sdata.points and not overwrite:
            raise ValueError(
                f"Tile {new_name} already exists, " f"set overwrite=True to continue."
            )
        tiles = self.get_tiles_table(name)[indices]
        if len(tiles) == 0:
            warnings.warn(f"No tiles can be created for {new_name}.")
            return
        spec = self.get_tile_spec(name)
        xy = tiles[["x", "y"]].values
        tissue_id = tiles["tissue_id"].values
        self.add_tiles(xy, tissue_id, new_name, spec)
        tiles_data = {
            key: tiles[key].values
            for key in tiles.columns
            if key not in ["x", "y", "tissue_id"]
        }
        self.add_tiles_data(tiles_data, new_name)

    def get_tiles_table(self, name):
        if name in self.sdata.tables:
            return self.sdata.tables[name].obs
        if f"{name}_table" in self.sdata.tables:
            return self.sdata.tables[f"{name}_table"].obs
        else:
            raise ValueError(f"Tile {name} not found.")

    def get_tile_spec(self, key) -> TileSpec:
        if f"{key}_table" not in self.sdata.tables:
            raise ValueError(f"Tile {key} not found.")
        return TileSpec(**self.sdata.tables[f"{key}_table"].uns["tile_spec"])

    def add_features(
        self,
        features: np.ndarray | pd.DataFrame,
        tile_name: str,
        feature_name: str,
        var=None,
    ):
        from spatialdata.models import TableModel

        # Check if the tile exists
        if tile_name not in self.sdata.points:
            raise ValueError(f"Tile {tile_name} not found.")
        # Check if the features are correct
        # if len(features) != self.sdata.tables[tile_name].shape[0]:
        #     raise ValueError(f"Features length does not match the tile {tile_name}.")
        points = self.sdata.tables[f"{tile_name}_table"].obs

        feature_adata = ad.AnnData(
            X=features,
            obs=points,
            obsm={"spatial": points[["x", "y"]].values},
            var=var,
        )
        feature_adata = TableModel.parse(
            feature_adata,
            region=[tile_name],
            region_key="tile_key",
            instance_key="id",
        )
        self.sdata.tables[f"{tile_name}_{feature_name}"] = feature_adata

    def add_slide_annotations(self, annotations: dict):
        if "annotations" in self.sdata.tables:
            self.sdata.tables["annotations"].uns["annotations"].update(annotations)
        self.sdata.tables["annotations"] = ad.AnnData(uns={"annotations": annotations})

    def get_slide_annotations(self):
        if "annotations" in self.sdata.tables:
            return self.sdata.tables["annotations"].uns["annotations"]
        return {}

    def write(self, file=None, overwrite=True, **kws):
        from uuid import uuid4
        import shutil

        if file is None:
            file = self.file

        # This is only a temporary solution
        # Need to wait for spatialdata to provide
        # a proper write method
        # 1. First write to a temporary file
        tmp_file = str(uuid4())
        self.sdata.write(tmp_file, overwrite=overwrite, **kws)
        # 2. Try to delete the original file
        try:
            shutil.rmtree(file)
        except FileNotFoundError:
            pass
        # 3. Move the temporary file to the original file
        shutil.move(tmp_file, file)

    def is_backed(self):
        return self.sdata.is_backed()


class WSI(SlideData):
    """Whole-slide image data

    Parameters
    ----------
    slide : Any
        The slide file or URL. You can use any file path or URL.
    backed_file : str, optional
        The backed file path, by default will create
        a zarr file with the same name as the slide file.
        You can either supply a file path or a directory.
    name : str, optional
        The name of the slide, by default will derive from the slide url.
    reader : str, optional, {"auto", "openslide", "tiffslide"}
        The reader type, by default "auto"
    cache_dir : str, optional
        The cache directory, by default None


    """

    def __init__(
        self,
        slide: Any,
        backed_file=None,
        name=None,
        reader="auto",
        cache_dir=None,
        **kwargs,
    ):
        # Check if the slide is a file or URL
        self.slide = str(slide)
        if name is None:
            self.name = Path(self.slide).stem
        else:
            self.name = name
        self.slide_origin = self.slide
        fs, slide_path = url_to_fs(self.slide)
        if not fs.exists(slide_path):
            raise ValueError(f"Slide {self.slide} not existed or not accessible.")
        # Early attempt with reader
        reader_cls = get_reader(reader)

        # Try to download remote slide when possible
        if not isinstance(fs, LocalFileSystem):
            # Download the slide to a temporary file
            if cache_dir is None:
                cache_dir = "TMP"
            cfs = WholeFileCacheFileSystem(
                fs=fs,
                cache_storage=cache_dir,
                same_names=True,
            )
            cache_slide = cfs.open(self.slide)
            self.slide = cache_slide.name
            self.slide_origin = cache_slide.original

        self.reader = reader_cls(self.slide, **kwargs)

        if backed_file is None:
            self.backed_file = Path(self.slide).with_suffix(".zarr")
        else:
            backed_file = Path(backed_file)
            if backed_file.is_dir():
                zarr_name = Path(self.slide).with_suffix(".zarr").name
                self.backed_file = backed_file / zarr_name
            else:
                self.backed_file = backed_file

        super().__init__(self.backed_file)

    def __repr__(self):
        return (
            f"Slide: {self.slide}\n"
            f"Backed Zarr: {self.backed_file}\n"
            f"Reader: {self.reader.name}\n"
            f"{self.sdata}"
        )

    # def _repr_html_(self):
    #     return (f"<h3>Slide: {self.slide}</h3>"
    #             f"<h3>Backed Zarr: {self.backed_file}</h3>"
    #             f"<h3>Reader: {self.reader.name}</h3>"
    #             f"{self.sdata}")

    @property
    def metadata(self):
        return self.reader.metadata

    def get_region(self, x, y, width, height, level=0, **kwargs):
        return self.reader.get_region(x, y, width, height, level=level, **kwargs)

    def get_features(self, feature_key, tile_key="tiles"):
        X, var = None, None
        if feature_key is not None:
            feature_tb = self.sdata.tables[f"{tile_key}_{feature_key}"]
            X = feature_tb.X
            var = feature_tb.var

        tile_adata = self.sdata.tables[f"{tile_key}_table"]
        obs = tile_adata.obs
        obs.index = obs.index.astype(str)
        spatial = obs[["x", "y"]].values

        adata = ad.AnnData(
            X=X,
            var=var,
            obs=obs,
            obsm={"spatial": spatial},
        )
        slide_annotations = self.get_slide_annotations()
        adata.uns = {
            "annotations": slide_annotations,
            "metadata": asdict(self.metadata),
        }
        adata.uns.update(tile_adata.uns)
        return adata
