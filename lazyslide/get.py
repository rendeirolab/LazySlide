from collections import namedtuple

import cv2
import numpy as np
import pandas as pd

TissueContour = namedtuple("TissueContour", ["tissue_id", "contour", "holes"])


# TODO: Return random tissue images
def tissue_contours(
    wsi,
    key="tissue",
    as_array: bool = False,
    shuffle: bool = False,
    seed: int = 0,
):
    """A generator to extract tissue contours from the WSI.

    Parameters
    ----------
    wsi : WSI
        The whole-slide image object.
    key : str, default: "tissue"
        The tissue key.
    as_array : bool, default: False
        Return the contour as an array.
        If False, the contour is returned as a shapely geometry.
    shuffle : bool, default: False
        If True, return tissue contour in random order.

    """
    if f"{key}_contours" not in wsi.sdata.shapes:
        raise ValueError(f"Contour {key}_contours not found.")
    contours = wsi.get_shape_table(f"{key}_contours")
    if f"{key}_holes" in wsi.sdata.shapes:
        holes = wsi.get_shape_table(f"{key}_holes")
    else:
        holes = None

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
            yield TissueContour(
                tissue_id=tissue_id,
                contour=np.array(cnt.geometry.exterior.coords, dtype=np.int32),
                holes=hs,
            )
        else:
            yield TissueContour(tissue_id=tissue_id, contour=cnt.geometry, holes=hs)


TissueImage = namedtuple("TissueImage", ["tissue_id", "x", "y", "image"])


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
        ix = tissue_contour.tissue_id
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
        yield TissueImage(tissue_id=ix, x=x, y=y, image=img)


TileImage = namedtuple("TileImage", ["id", "x", "y", "tissue_id", "image"])


def tile_images(
    wsi,
    tile_key="tiles",
    raw=False,
    color_norm: str = None,
    shuffle: bool = False,
    sample_n: int = None,
    seed: int = 0,
):
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
    color_norm : str, {"macenko", "reinhard"}, default: None
        Color normalization method.
    shuffle : bool, default: False
        If True, return tile images in random order.
    sample_n : int, default: None
        The number of samples to return.
    seed : int, default: 0
        The random seed.

    """
    tile_spec = wsi.get_tile_spec(tile_key)
    # Check if the image needs to be transformed
    need_transform = (
        tile_spec.raw_width != tile_spec.width
        or tile_spec.raw_height != tile_spec.height
    )

    if color_norm is not None:
        from lazyslide.cv.colornorm import ColorNormalizer

        cn = ColorNormalizer(method=color_norm)
        cn_func = lambda x: cn(x).numpy()  # noqa
    else:
        cn_func = lambda x: x  # noqa

    points = wsi.get_tiles_table(tile_key)
    if sample_n is not None:
        points = points.sample(n=sample_n, random_state=seed)
    elif shuffle:
        points = points.sample(frac=1, random_state=seed)

    for _, row in points.iterrows():
        x = row["x"]
        y = row["y"]
        ix = row["id"]
        tix = row["tissue_id"]
        img = wsi.reader.get_region(
            x, y, tile_spec.raw_width, tile_spec.raw_height, level=tile_spec.level
        )
        img = cn_func(img)
        if raw and not need_transform:
            yield TileImage(id=ix, x=x, y=y, tissue_id=tix, image=img)
        else:
            yield TileImage(
                id=ix,
                x=x,
                y=y,
                tissue_id=tix,
                image=cv2.resize(img, (tile_spec.width, tile_spec.height)),
            )


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


def features_anndata(
    wsi,
    tile_key="tiles",
    feature_key=None,
):
    """Convert the WSI to an AnnData object"""
    return wsi.get_features(feature_key, tile_key=tile_key)


def tiles_table(wsi, key="tiles"):
    """Return the tile table"""
    return wsi.get_tiles_table(key)


def shape_table(wsi, key="tissue_contours"):
    """Return the shape table"""
    return wsi.get_shape_table(key)


def n_tissue(wsi, key="tissue"):
    """Return the number of tissue regions"""
    cnt = wsi.sdata.shapes[f"{key}_contours"]
    return cnt["tissue_id"].nunique()


def n_tiles(wsi, key="tiles"):
    """Return the number of tiles"""
    return len(wsi.sdata.points[key])
