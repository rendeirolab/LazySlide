from __future__ import annotations

import json
import warnings
from collections import namedtuple
from dataclasses import dataclass, asdict
from functools import cached_property
from typing import Mapping, Optional, List, Sequence

import cv2
import numpy as np
import pandas as pd
import zarr
from ome_zarr.io import parse_url
from anndata import AnnData
from spatialdata import SpatialData

from wsi_data.reader import ReaderBase


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

    def to_dict(self):
        return asdict(self)

    def to_json(self):
        return json.dumps(asdict(self))


class WSIData(object):
    """WSI Data

    WSI data is a container class (SpatialData + Reader)
    to fit the use of whole slide images.

    The wsi data will be initialized with a reader object that can
    operate the whole slide image file, by default, the whole slide image
    will not be attached to the SpatialData. However, for visualization purpose,
    a thumbnail version of the whole slide image can be attached.

    There are three main components in the SlideData:

    - Whole slide image (WSI)
    - Tissues contours (shapes)
    - Tiles locations (shapes)
    - Features (images)

    """

    TILE_SPEC_KEY = "tile_spec"
    SLIDE_PROPERTIES_KEY = "slide_properties"

    def __init__(self, reader: ReaderBase, sdata: SpatialData, backed_file):
        self._reader = reader
        self._sdata = sdata
        self._backed_file = backed_file
        self._write_elements = set()

        if self.SLIDE_PROPERTIES_KEY not in sdata:
            sdata.tables[self.SLIDE_PROPERTIES_KEY] = AnnData(
                uns=reader.properties.to_dict()
            )
            self._write_elements.add(self.SLIDE_PROPERTIES_KEY)

    def __repr__(self):
        return (
            f"WSI: {self.reader.file}\n"
            f"Reader: {self.reader.name}\n"
            f"{self.sdata.__repr__()}"
        )

    def __getitem__(self, item):
        return self.sdata.__getitem__(item)

    def add_write_elements(self, name: str | Sequence[str]):
        if isinstance(name, str):
            self._write_elements.add(name)
        else:
            self._write_elements.update(name)

    @property
    def reader(self):
        return self._reader

    @property
    def sdata(self):
        return self._sdata

    @property
    def properties(self):
        return self.reader.properties

    def n_tissue(self, key):
        return len(self.sdata.shapes[key])

    def n_tiles(self, key):
        return self.n_tissue(key)

    def tile_spec(self, key) -> TileSpec:
        if self.TILE_SPEC_KEY in self.sdata:
            spec = self.sdata.tables[self.TILE_SPEC_KEY].uns[key]
            return TileSpec(**spec)

    def set_mpp(self, mpp):
        # TODO: Allow user to set the mpp of slide
        pass

    def read_region(self, x, y, width, height, level=0, **kwargs):
        return self.reader.get_region(x, y, width, height, level=level, **kwargs)

    def add_tissues(self, key, tissues, ids=None, **kws):
        import geopandas as gpd
        from shapely.geometry import Polygon
        from spatialdata.models import ShapesModel

        if ids is None:
            ids = np.arange(len(tissues))
        gdf = gpd.GeoDataFrame(
            data={"tissue_id": ids, "geometry": [Polygon(c) for c in tissues]}
        )
        cs = ShapesModel.parse(gdf, **kws)
        self.sdata.shapes[key] = cs
        self._write_elements.add(key)

    def add_tiles(self, key, xys, tile_spec, tissue_ids, **kws):
        import geopandas as gpd
        from shapely.geometry import Polygon
        import anndata as ad
        from spatialdata.models import ShapesModel

        # Tiles should be stored as polygon
        # This will allow easy query of which cells in which tiles
        w, h = tile_spec.raw_width, tile_spec.raw_height
        gdf = gpd.GeoDataFrame(
            data={
                "id": np.arange(len(xys)),
                "x": xys[:, 0],
                "y": xys[:, 1],
                "tissue_id": tissue_ids,
                "geometry": [
                    Polygon([(x, y), (x, y + h), (x + w, y + h), (x + w, y)])
                    for (x, y) in xys
                ],
            }
        )
        cs = ShapesModel.parse(gdf, **kws)
        self.sdata.shapes[key] = cs

        if self.TILE_SPEC_KEY in self.sdata.tables:
            spec_data = self.sdata.tables[self.TILE_SPEC_KEY]
        else:
            spec_data = ad.AnnData()
        spec_data.uns[key] = tile_spec.to_dict()
        self.sdata.tables[self.TILE_SPEC_KEY] = spec_data

        self._write_elements.add(key)
        self._write_elements.add(self.TILE_SPEC_KEY)

    def update_shapes_data(self, key, data):
        shapes = self.sdata.shapes[key]
        if isinstance(data, Mapping):
            self.sdata.shapes[key] = shapes.assign(**data)
        elif isinstance(data, pd.DataFrame):
            self.sdata.shapes[key] = pd.concat(
                [shapes, data], axis=1, ignore_index=True
            )
        self._write_elements.add(key)

    def add_features(self, key, features, **kws):
        from spatialdata.models import Labels2DModel

        f_img = Labels2DModel.parse(features, dims=("y", "x"), **kws)
        self.sdata.labels[key] = f_img
        self._write_elements.add(key)

    def save(self, consolidate_metadata: bool = True):
        # Create the store first
        store = parse_url(self._backed_file, mode="w").store
        _ = zarr.group(store=store, overwrite=True)
        store.close()

        # Assign to SpatialData
        self.sdata.path = self._backed_file

        self.sdata.write_element(list(self._write_elements), overwrite=True)
        if consolidate_metadata:
            self.sdata.write_consolidated_metadata()

    @cached_property
    def iter(self) -> SlideDataIterAccessor:
        return SlideDataIterAccessor(self)

    @cached_property
    def get(self) -> SlideDataGetAccessor:
        return SlideDataGetAccessor(self)

    def _check_feature_key(self, feature_key, tile_key=None):
        msg = f"{feature_key} doesn't exist"
        if feature_key in self.sdata:
            return feature_key
        else:
            if tile_key is not None:
                feature_key = f"{tile_key}_{feature_key}"
                if feature_key in self.sdata:
                    return feature_key
                msg = f"Neither {feature_key} or {tile_key}_{feature_key} exist"

        raise KeyError(msg)


class SlideDataIterAccessor(object):
    TissueContour = namedtuple("TissueContour", ["tissue_id", "contour", "holes"])
    TissueImage = namedtuple("TissueImage", ["tissue_id", "x", "y", "image", "mask"])
    TileImage = namedtuple("TileImage", ["id", "x", "y", "tissue_id", "image"])

    def __init__(self, obj: WSIData):
        self._obj = obj

    def tissue_contours(
        self,
        key,
        as_array: bool = False,
        shuffle: bool = False,
        seed: int = 0,
    ):
        """A generator to extract tissue contours from the WSI.

        Parameters
        ----------
        key : str
            The tissue key.
        as_array : bool, default: False
            Return the contour as an array.
            If False, the contour is returned as a shapely geometry.
        shuffle : bool, default: False
            If True, return tissue contour in random order.
        seed : int, default: 0


        """
        holes_key = f"{key}_holes"

        contours = self._obj.sdata.shapes[key]
        if holes_key not in self._obj.sdata:
            holes = None
        else:
            holes = self._obj.sdata.shapes[holes_key]

        if shuffle:
            contours = contours.sample(frac=1, random_state=seed)

        for ix, cnt in contours.iterrows():
            tissue_id = cnt["tissue_id"]
            if holes is not None:
                hs = holes[holes["tissue_id"] == tissue_id].geometry.tolist()
                if as_array:
                    hs = [np.array(h.exterior.coords, dtype=np.int32) for h in hs]
            else:
                hs = []
            if as_array:
                yield self.TissueContour(
                    tissue_id=tissue_id,
                    contour=np.array(cnt.geometry.exterior.coords, dtype=np.int32),
                    holes=hs,
                )
            else:
                yield self.TissueContour(
                    tissue_id=tissue_id, contour=cnt.geometry, holes=hs
                )

    def tissue_images(
        self,
        key,
        level=0,
        mask_bg=False,
        tissue_mask=False,
        color_norm: str = None,
    ):
        """Extract tissue images from the WSI.

        Parameters
        ----------
        key : str
            The tissue key.
        level : int, default: 0
            The level to extract the tissue images.
        mask_bg : bool | int, default: False
            Mask the background with the given value.
            If False, the background is not masked.
            If True, the background is masked with 0.
            If an integer, the background is masked with the given value.
        color_norm : str, {"macenko", "reinhard"}, default: None
            Color normalization method.

        """
        import cv2

        level_downsample = self._obj.properties.level_downsample[level]
        if color_norm is not None:
            from lazyslide_cv.colornorm import ColorNormalizer

            cn = ColorNormalizer(method=color_norm)
            cn_func = lambda x: cn(x).numpy()  # noqa
        else:
            cn_func = lambda x: x  # noqa

        if isinstance(mask_bg, bool):
            do_mask = mask_bg
            if do_mask:
                mask_bg = 0
        else:
            do_mask = True
            mask_bg = mask_bg
        for tissue_contour in self.tissue_contours(key):
            ix = tissue_contour.tissue_id
            contour = tissue_contour.contour
            holes = tissue_contour.holes
            minx, miny, maxx, maxy = contour.bounds
            x = int(minx)
            y = int(miny)
            w = int(maxx - minx) / level_downsample
            h = int(maxy - miny) / level_downsample
            img = self._obj.reader.get_region(x, y, w, h, level=level)
            img = cn_func(img)

            mask = None
            if do_mask or tissue_mask:
                mask = np.zeros_like(img[:, :, 0])
                # Offset and scale the contour
                offset_x, offset_y = x / level_downsample, y / level_downsample
                coords = np.array(contour.exterior.coords) - [offset_x, offset_y]
                coords = (coords / level_downsample).astype(np.int32)
                # Fill the contour with 1
                cv2.fillPoly(mask, [coords], 1)

                # Fill the holes with 0
                for hole in holes:
                    hole = np.array(hole.exterior.coords) - [offset_x, offset_y]
                    hole = (hole / level_downsample).astype(np.int32)
                    cv2.fillPoly(mask, [hole], 0)

            if do_mask:
                # Fill everything that is not the contour
                # (which is background) with 0
                img[mask != 1] = mask_bg
            if not tissue_mask:
                mask = None
            else:
                mask = mask.astype(bool)

            yield self.TissueImage(tissue_id=ix, x=x, y=y, image=img, mask=mask)

    def tile_images(
        self,
        key,
        raw=False,
        color_norm: str = None,
        shuffle: bool = False,
        sample_n: int = None,
        seed: int = 0,
    ):
        """Extract tile images from the WSI.

        Parameters
        ----------
        key : str
            The tile key.
        raw : bool, default: True
            Return the raw image without resizing.
            If False, the image is resized to the requested tile size.
        color_norm : str, {"macenko", "reinhard"}, default: None
            Color normalization method.
        shuffle : bool, default: False
            If True, return tile images in random order.
        sample_n : int, default: None
            The number of samples to return.
        seed : int, default: 0
            The random seed.

        """
        tile_spec = self._obj.tile_spec(key)
        # Check if the image needs to be transformed
        need_transform = (
            tile_spec.raw_width != tile_spec.width
            or tile_spec.raw_height != tile_spec.height
        )

        if color_norm is not None:
            from lazyslide_cv.colornorm import ColorNormalizer

            cn = ColorNormalizer(method=color_norm)
            cn_func = lambda x: cn(x).numpy()  # noqa
        else:
            cn_func = lambda x: x  # noqa

        points = self._obj.sdata[key]
        if sample_n is not None:
            points = points.sample(n=sample_n, random_state=seed)
        elif shuffle:
            points = points.sample(frac=1, random_state=seed)

        for _, row in points.iterrows():
            x = row["x"]
            y = row["y"]
            ix = row["id"]
            tix = row["tissue_id"]
            img = self._obj.reader.get_region(
                x, y, tile_spec.raw_width, tile_spec.raw_height, level=tile_spec.level
            )
            img = cn_func(img)
            if raw and not need_transform:
                yield self.TileImage(id=ix, x=x, y=y, tissue_id=tix, image=img)
            else:
                yield self.TileImage(
                    id=ix,
                    x=x,
                    y=y,
                    tissue_id=tix,
                    image=cv2.resize(img, (tile_spec.width, tile_spec.height)),
                )


class SlideDataGetAccessor(object):
    def __init__(self, obj: WSIData):
        self._obj = obj

    def pyramids(self):
        heights, widths = zip(*self._obj.properties.level_shape)
        return pd.DataFrame(
            {
                "height": heights,
                "width": widths,
                "downsample": self._obj.properties.level_downsample,
            },
            index=pd.RangeIndex(self._obj.properties.n_level, name="level"),
        )

    def features_anndata(self, tile_key, feature_key):
        import anndata as ad

        feature_key = self._obj._check_feature_key(feature_key, tile_key)
        X = self._obj.sdata.labels[feature_key].values
        tile_table = self._obj.sdata.shapes[tile_key]
        tile_xy = tile_table[["x", "y"]].values
        obs = tile_table.drop(columns=["geometry"])
        # To suppress anndata warning
        obs.index = obs.index.astype(str)
        return ad.AnnData(
            X=X,
            obs=obs,
            obsm={"spatial": tile_xy},
            uns={
                "tile_spec": self._obj.tile_spec(tile_key),
                "slide_properties": self._obj.properties.to_dict(),
            },
        )
