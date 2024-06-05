import cv2
import numpy as np
import pandas as pd


def tissue_images(wsi, tissue_key="tissue", level=0, mask_bg=False):
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

    """
    if f"{tissue_key}_contours" not in wsi.sdata.shapes:
        raise ValueError(f"Contour {tissue_key}_contours not found.")
    contours = wsi.sdata.shapes[f"{tissue_key}_contours"]
    tissue_bboxes = contours.bounds
    level_downsample = wsi.metadata.level_downsample[level]
    if isinstance(mask_bg, bool):
        do_mask = mask_bg
        if do_mask:
            mask_bg = 0
    else:
        do_mask = True
        mask_bg = mask_bg
    for ix, (minx, miny, maxx, maxy) in tissue_bboxes.iterrows():
        x = int(minx)
        y = int(miny)
        w = int(maxx - minx) / level_downsample
        h = int(maxy - miny) / level_downsample
        img = wsi.reader.get_region(x, y, w, h, level=level)
        if do_mask:
            cnt = contours.geometry[ix]
            mask = np.zeros_like(img[:, :, 0])
            # Offset and scale the contour
            offset_x, offset_y = x / level_downsample, y / level_downsample
            coords = np.array(cnt.exterior.coords) - [offset_x, offset_y]
            coords = (coords / level_downsample).astype(np.int32)
            # Fill the contour with 1
            cv2.fillPoly(mask, [coords], 1)
            # Fill everything that is not the contour
            # (which is background) with 0
            img[mask != 1] = mask_bg
        yield (x, y), img


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


# def to_anndata(wsi,
#                tile_key="tiles",
#                feature_key=None,
#                ):
#     """Convert the WSI to an AnnData object"""
#     import anndata as ad
#     adata = ad.AnnData(X=wsi.sdata.points["tiles"].to_pandas())
#     slide_metadata = wsi.sdata.tables["metadata"].uns["metadata"]
#     slide_properties = wsi.metadata
#     adata.uns = {"metadata": slide_metadata, "properties": slide_properties}
#     for key, table in wsi.sdata.tables.items():
#         if key == "metadata":
#             continue
#         adata.uns[key] = table.to_pandas()
#     return adata
