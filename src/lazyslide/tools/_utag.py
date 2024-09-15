import numpy as np

from lazyslide._const import Key
from wsidata import WSIData


def utag_feature(
    wsi: WSIData,
    feature_key: str,
    tile_key: str = Key.tiles,
):
    # Get the spatial connectivity
    try:
        A = wsi.sdata.tables[f"{tile_key}_graph"].obsp["spatial_connectivities"]
    except KeyError:
        raise ValueError("Please run `tile_graph` before using `utag_feature`")
    A = A + np.eye(A.shape[0])
    # L1 norm for each row
    norms = np.sum(np.abs(A), axis=1)
    # Normalize the array
    A_norm = A / norms

    feature_key = wsi._check_feature_key(feature_key, tile_key)
    feature_X = wsi.sdata.tables[feature_key].X
    A_spatial = np.transpose(feature_X) @ A_norm
    A_spatial = np.transpose(A_spatial)
    wsi.sdata.tables[feature_key].layers["utag"] = np.asarray(A_spatial)
    wsi.add_write_elements(feature_key)
