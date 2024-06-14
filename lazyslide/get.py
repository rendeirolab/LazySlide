from collections import namedtuple
from typing import Literal

import cv2
import numpy as np
import pandas as pd

TissueContour = namedtuple("TissueContour", ["id", "contour", "holes"])


def tissue_contours(
    wsi,
    key="tissue",
    return_type: Literal["shapely", "numpy"] = "shapely",
):
    if f"{key}_contours" not in wsi.sdata.shapes:
        raise ValueError(f"Contour {key}_contours not found.")
    contours = wsi.sdata.shapes[f"{key}_contours"]
    if f"{key}_holes" in wsi.sdata.shapes:
        holes = wsi.sdata.shapes[f"{key}_holes"]
    else:
        holes = None

    for ix, cnt in contours.iterrows():
        tissue_id = cnt["tissue_id"]
        if holes is not None:
            hs = holes[holes["tissue_id"] == tissue_id].geometry.tolist()
            if return_type == "numpy":
                hs = [np.array(h.exterior.coords, dtype=np.int32) for h in hs]
        else:
            hs = []
        if return_type == "numpy":
            yield TissueContour(
                id=tissue_id,
                contour=np.array(cnt.geometry.exterior.coords, dtype=np.int32),
                holes=hs,
            )
        else:
            yield TissueContour(id=tissue_id, contour=cnt.geometry, holes=hs)


TissueImage = namedtuple("TissueImage", ["id", "x", "y", "image"])


def tissue_images(
    wsi,
    tissue_key="tissue",
    level=0,
    mask_bg=False,
    color_norm: str = None,
):
    """Extract tissue images from the WSI.

    Parameters
    ----------
    wsi : WSI
        The whole-slide image object.
    tissue_key : str, default: "tissue"
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

    level_downsample = wsi.metadata.level_downsample[level]
    if color_norm is not None:
        from lazyslide.cv.colornorm import ColorNormalizer

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
    for tissue_contour in tissue_contours(wsi, key=tissue_key):
        ix = tissue_contour.id
        contour = tissue_contour.contour
        holes = tissue_contour.holes
        minx, miny, maxx, maxy = contour.bounds
        x = int(minx)
        y = int(miny)
        w = int(maxx - minx) / level_downsample
        h = int(maxy - miny) / level_downsample
        img = wsi.reader.get_region(x, y, w, h, level=level)
        img = cn_func(img)
        if do_mask:
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
            # Fill everything that is not the contour
            # (which is background) with 0
            img[mask != 1] = mask_bg
        yield TissueImage(id=ix, x=x, y=y, image=img)


def tile_images(wsi, tile_key="tiles", raw=True):
    """Extract tile images from the WSI.

    Parameters
    ----------
    wsi : WSI
        The whole-slide image object.
    tile_key : str, default: "tiles"
        The tile key.
    raw : bool, default: True
        Return the raw image without resizing.
        If False, the image is resized to the requested tile size.

    """
    if tile_key not in wsi.sdata.points:
        raise ValueError(f"Tile {tile_key} not found.")
    tile_spec = wsi.get_tile_spec(tile_key)
    # Check if the image needs to be transformed
    need_transform = (
        tile_spec.ops_width != tile_spec.width
        or tile_spec.ops_height != tile_spec.height
    )
    for x, y in wsi.sdata.points[tile_key][["x", "y"]].compute().to_numpy():
        img = wsi.reader.get_region(
            x, y, tile_spec.ops_width, tile_spec.ops_height, level=tile_spec.level
        )
        if raw and not need_transform:
            yield (x, y), img
        else:
            yield (x, y), cv2.resize(img, (tile_spec.width, tile_spec.height))


def pyramids(wsi):
    """Return a dataframe with the pyramid levels"""
    heights, widths = zip(*wsi.metadata.level_shape)
    return pd.DataFrame(
        {
            "height": heights,
            "width": widths,
            "downsample": wsi.metadata.level_downsample,
        },
        index=pd.RangeIndex(wsi.metadata.n_level, name="level"),
    )


def tiles_anndata(
    wsi,
    tile_key="tiles",
    feature_key=None,
):
    """Convert the WSI to an AnnData object"""
    import anndata as ad

    X, var = None, None
    if feature_key is not None:
        feature_tb = wsi.sdata.tables[f"{tile_key}/{feature_key}"]
        X = feature_tb.X
        var = feature_tb.var

    obs = wsi.sdata.points[tile_key].compute()
    obs.index = obs.index.astype(str)
    spatial = obs[["x", "y"]].values

    adata = ad.AnnData(
        X=X,
        var=var,
        obs=obs,
        obsm={"spatial": spatial},
    )
    if "annotations" in wsi.sdata.tables:
        slide_annotations = wsi.sdata.tables["annotations"].uns["annotations"]
    else:
        slide_annotations = {}
    adata.uns = {
        "annotations": slide_annotations,
        "metadata": wsi.metadata.model_dump(),
    }
    for key, table in wsi.sdata.tables.items():
        if key == f"{tile_key}_spec":
            adata.uns[key] = table.uns["tiles_spec"]
    return adata


def n_tissue(wsi, key="tissue"):
    """Return the number of tissue regions"""
    cnt = wsi.sdata.shapes[f"{key}_contours"]
    return cnt["tissue_id"].nunique()


def n_tiles(wsi, key="tiles"):
    """Return the number of tiles"""
    return wsi.sdata.points[key].shape[0].compute()
