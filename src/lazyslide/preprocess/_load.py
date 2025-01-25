from __future__ import annotations

from pathlib import Path
from typing import List, Callable

from geopandas import GeoDataFrame
from shapely import Polygon

from lazyslide._const import Key
from wsidata import WSIData
from wsidata.io import update_shapes_data, add_shapes


def load_annotations(
    wsi: WSIData,
    annotations: str | Path | GeoDataFrame = None,
    explode: bool = True,
    in_bounds: bool = False,
    join_with: str | List[str] = Key.tissue,
    join_to: str = None,
    min_area: float = 1e2,
    key_added: str = "annotations",
):
    """Load the annotation file and add it to the WSIData

    Parameters
    ----------
    wsi : :class:`WSIData <wsidata.WSIData>`
        The WSIData object to work on.
    annotations : str, Path, GeoDataFrame
        The path to the annotation file or the GeoDataFrame.
    explode : bool, default: True
        Whether to explode the annotations.
    in_bounds : bool, default: False
        Whether to move the annotations to the slide bounds.
    join_with : str, List[str], default: 'tissues'
        The key to join the annotations with.
    join_to : str, default: None
        The key to join the annotations to.
    min_area : float, default: 1e2
        The minimum area of the annotation.
    key_added : str, default: 'annotations'
        The key to store the annotations.

    """
    import geopandas as gpd

    if isinstance(annotations, (str, Path)):
        geo_path = Path(annotations)
        anno_df = gpd.read_file(geo_path)
    elif isinstance(annotations, GeoDataFrame):
        anno_df = annotations
    else:
        raise ValueError(f"Invalid annotations: {annotations}")

    # remove crs
    anno_df.crs = None

    if explode:
        anno_df = (
            anno_df.explode()
            .assign(**{"__area__": lambda x: x.geometry.area})
            .query(f"__area__ > {min_area}")
            .drop(columns=["__area__"], errors="ignore")
            .reset_index(drop=True)
        )

    if in_bounds:
        from functools import partial
        from shapely.affinity import translate

        xoff, yoff, _, _ = wsi.properties.bounds
        trans = partial(translate, xoff=xoff, yoff=yoff)
        anno_df["geometry"] = anno_df["geometry"].apply(lambda x: trans(x))

    # get tiles
    if isinstance(join_with, str):
        join_with = [join_with]

    join_anno_df = anno_df.copy()
    for key in join_with:
        if key in wsi:
            shapes_df = wsi[key]
            # join the annotations with the tiles
            join_anno_df = (
                gpd.sjoin(shapes_df, join_anno_df, how="right", predicate="intersects")
                .reset_index(drop=True)
                .drop(columns=["index_left"], errors="ignore")
            )
    add_shapes(wsi, key_added, join_anno_df)

    # TODO: still Buggy
    if join_to is not None:
        if join_to in wsi:
            shapes_df = wsi[join_to]
            # join the annotations with the tiles
            shapes_df = (
                gpd.sjoin(
                    shapes_df[["geometry"]], anno_df, how="left", predicate="intersects"
                )
                .reset_index(drop=True)
                .drop(columns=["index_right"], errors="ignore")
            )
            update_shapes_data(wsi, join_to, shapes_df)
