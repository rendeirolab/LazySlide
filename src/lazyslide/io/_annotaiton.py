from __future__ import annotations

import json
from itertools import cycle
from pathlib import Path
from typing import List, Literal, Mapping, Iterable

import pandas as pd
from geopandas import GeoDataFrame
from wsidata import WSIData
from wsidata.io import update_shapes_data, add_shapes

from lazyslide._const import Key


def _in_bounds_transform(wsi: WSIData, annos: GeoDataFrame, reverse: bool = False):
    from functools import partial
    from shapely.affinity import translate

    xoff, yoff, _, _ = wsi.properties.bounds
    if reverse:
        xoff, yoff = -xoff, -yoff
    trans = partial(translate, xoff=xoff, yoff=yoff)
    annos["geometry"] = annos["geometry"].apply(lambda x: trans(x))
    return annos


def load_annotations(
    wsi: WSIData,
    annotations: str | Path | GeoDataFrame = None,
    *,
    explode: bool = True,
    in_bounds: bool = False,
    join_with: str | List[str] = Key.tissue,
    join_to: str = None,
    json_flatten: str | List[str] = "classification",
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
    json_flatten : str, default: "classification"
        The column(s) to flatten the json data, if not exist, it will be ignored.
        "classification" is the default column for the QuPath annotations.
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

    if json_flatten is not None:

        def flatten_json(x):
            if isinstance(x, dict):
                return x
            elif isinstance(x, str):
                try:
                    return json.loads(x)
                except json.JSONDecodeError:
                    return {}

        if isinstance(json_flatten, str):
            json_flatten = [json_flatten]
        for col in json_flatten:
            if col in anno_df.columns:
                anno_df[col] = anno_df[col].apply(flatten_json)
                anno_df = anno_df.join(
                    anno_df[col].apply(pd.Series).add_prefix(f"{col}_")
                )
                anno_df.drop(columns=[col], inplace=True)

    if in_bounds:
        anno_df = _in_bounds_transform(wsi, anno_df)

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
                .drop(columns=["index_left"])
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


def export_annotations(
    wsi: WSIData,
    key: str,
    *,
    in_bounds: bool = False,
    classes: str = None,
    colors: str | Mapping = None,
    format: Literal["qupath"] = "qupath",
    file: str | Path = None,
):
    """
    Export the annotations

    Parameters
    ----------
    wsi : :class:`WSIData <wsidata.WSIData>`
        The WSIData object to work on.
    key : str
        The key to export.
    in_bounds : bool, default: False
        Whether to move the annotations to the slide bounds.
    classes : str, default: None
        The column to use for the classification.
        If None, the classification will be ignored.
    colors : str, Mapping, default: None
        The column to use for the color.
        If None, the color will be ignored.
    format : str, default: 'qupath'
        The format to export.
        Currently only 'qupath' is supported.
    file : str, Path, default: None
        The file to save the annotations.
        If None, the annotations will not be saved.


    """
    gdf = wsi.shapes[key].copy()
    if in_bounds:
        gdf = _in_bounds_transform(wsi, gdf, reverse=True)

    if format == "qupath":
        # Prepare classification column
        import json

        if classes is not None:
            class_values = gdf[classes]

            if colors is None:
                # Assign default colors
                colors = cycle(
                    [
                        "#1B9E77",  # Teal Green
                        "#D95F02",  # Burnt Orange
                        "#7570B3",  # Deep Lavender
                        "#E7298A",  # Magenta
                        "#66A61E",  # Olive Green
                        "#E6AB02",  # Goldenrod
                        "#A6761D",  # Earthy Brown
                        "#666666",  # Charcoal Gray
                        "#1F78B4",  # Cool Blue
                    ]
                )

            if colors is not None:
                color_values = cycle([])
                if isinstance(colors, str):
                    color_values = gdf[colors]
                elif isinstance(colors, Iterable):
                    # if sequence of colors, map to class values
                    colors = dict(zip(pd.unique(class_values), colors))
                else:
                    raise ValueError(f"Invalid colors: {colors}")

                if isinstance(colors, Mapping):
                    color_values = map(lambda x: colors.get(x, None), gdf[classes])

                # covert color to rgb array
                from matplotlib.colors import to_rgb

                color_values = map(
                    lambda x: tuple(int(255 * c) for c in to_rgb(x))
                    if x is not None
                    else None,
                    color_values,
                )

            classifications = []
            for class_value, color_value in zip(class_values, color_values):
                json_string = json.dumps({"name": class_value, "color": color_value})
                classifications.append(json_string)
            gdf["classification"] = classifications

    if file is not None:
        gdf.to_file(file)

    return gdf
