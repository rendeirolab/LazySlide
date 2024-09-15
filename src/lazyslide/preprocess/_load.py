from __future__ import annotations

from pathlib import Path
from typing import List, Callable

from geopandas import GeoDataFrame

from lazyslide._const import Key
from wsidata import WSIData


def load_annotations(
    wsi: WSIData,
    annotations: str | Path | GeoDataFrame = None,
    in_bounds: bool = False,
    join_with: str | List[str] = Key.tiles,
    key_added: str = "annotations",
):
    """Load the geojson file and add it to the WSI data"""
    import geopandas as gpd

    if isinstance(annotations, (str, Path)):
        geo_path = Path(annotations)
        anno_df = gpd.read_file(geo_path)
    elif isinstance(annotations, GeoDataFrame):
        anno_df = annotations
    else:
        raise ValueError(f"Invalid annotations: {annotations}")

    if in_bounds:
        from functools import partial
        from shapely.affinity import translate

        xoff, yoff, _, _ = wsi.properties.bounds
        trans = partial(translate, xoff=xoff, yoff=yoff)
        anno_df["geometry"] = anno_df["geometry"].apply(lambda x: trans(x))

    wsi.add_shapes(key_added, anno_df)

    # get tiles
    if isinstance(join_with, str):
        join_with = [join_with]

    for key in join_with:
        if key in wsi.sdata:
            tile_df = wsi.sdata[key]
            # join the annotations with the tiles
            gdf = gpd.sjoin(tile_df[["geometry"]], anno_df, how="left", op="intersects")
            wsi.update_shapes_data(key, gdf)
    return wsi
