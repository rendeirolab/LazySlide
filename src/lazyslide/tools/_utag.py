import numpy as np
from wsidata import WSIData

from lazyslide._const import Key


def feature_utag(
    wsi: WSIData,
    feature_key: str,
    tile_key: str = Key.tiles,
    graph_key: str = None,
):
    """
    Transform feature with UTAG.

    Parameters
    ----------
    wsi: :class:`WSIData <wsidata.WSIData>`
        The WSIData object.
    feature_key: str
        The feature key.
    tile_key: str, default: 'tiles'
        The tile key.
    graph_key: str
        The graph key.

    """
    # Get the spatial connectivity
    try:
        if graph_key is None:
            graph_key = f"{tile_key}_graph"
        A = wsi.tables[graph_key].obsp["spatial_connectivities"]
    except KeyError:
        raise ValueError(
            "The tile graph is needed to transform feature with UTAG, Please run `pp.tile_graph` first."
        )
    A = A + np.eye(A.shape[0])
    # L1 norm for each row
    norms = np.sum(np.abs(A), axis=1)
    # Normalize the array
    A_norm = A / norms

    feature_key = wsi._check_feature_key(feature_key, tile_key)
    feature_X = wsi.tables[feature_key].X
    A_spatial = np.transpose(feature_X) @ A_norm
    A_spatial = np.transpose(A_spatial)
    wsi.tables[feature_key].layers["utag"] = np.asarray(A_spatial)
