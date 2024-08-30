import numpy as np

from lazyslide._const import Key
from wsi_data import WSIData


def utag_feature(
    wsi: WSIData,
    feature_key: str,
    tile_key: str = Key.tiles,
):
    # Get the spatial connectivity
    try:
        A = wsi.sdata.labels[f"{tile_key}_distances"].data
    except KeyError:
        raise ValueError("Please run `tile_graph` before using `prepare_utag`")
    A = A + np.eye(A.shape[0])
    # L1 norm for each row
    norms = np.sum(np.abs(A), axis=1, keepdims=True)
    # Normalize the array
    A_norm = A / norms

    feature_key = wsi._check_feature_key(feature_key, tile_key)
    feature_X = wsi.sdata.labels[feature_key].values
    A_spatial = np.transpose(feature_X) @ A_norm
    A_spatial = np.transpose(A_spatial)

    wsi.add_features(f"{feature_key}_utag", A_spatial, dims=("y", "x"))
