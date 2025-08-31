import warnings

import numpy as np
from wsidata import WSIData

from lazyslide._const import Key
from lazyslide._utils import find_stack_level


def spatial_features(
    wsi: WSIData,
    feature_key: str,
    method: str = "smoothing",
    tile_key: str = Key.tiles,
    graph_key: str = None,
    layer_key: str = "spatial_features",
):
    """
    Integrate spatial tile context with vision features using spatial feature smoothing.

    Parameters
    ----------
    wsi : :class:`WSIData <wsidata.WSIData>`
        The WSIData object.
    feature_key : str
        The feature key.
    method : str, default: 'smoothing'
        The method used for spatial feature smoothing. Currently, only 'smoothing' is supported.
    tile_key : str, default: 'tiles'
        The key of the tiles in the :bdg-danger:`shapes` slot.
    graph_key : str, optional
        The graph key. If None, defaults to '{tile_key}_graph'.
    layer_key : str, default: 'spatial_features'
        The key for the output layer in the feature table.

    Returns
    -------
    None.
        The transformed feature will be added to the :code:`spatial_features` layer of the feature AnnData.

    Examples
    --------

    .. code-block:: python

        >>> import lazyslide as zs
        >>> wsi = zs.datasets.sample()
        >>> zs.pp.find_tissues(wsi)
        >>> zs.pp.tile_tissues(wsi, 256, mpp=0.5)
        >>> zs.tl.feature_extraction(wsi, "resnet50")
        >>> zs.pp.tile_graph(wsi)
        >>> zs.tl.spatial_features(wsi, "resnet50")
        >>> wsi["resnet50"].layers["spatial_features"]

    """
    if method != "smoothing":
        raise ValueError(
            f"Unknown method '{method}'. Only 'smoothing' is currently supported."
        )

    # Get the spatial connectivity
    try:
        if graph_key is None:
            graph_key = f"{tile_key}_graph"
        A = wsi.tables[graph_key].obsp["spatial_connectivities"]
    except KeyError:
        raise ValueError(
            "The tile graph is needed to transform feature with spatial smoothing. Please run `pp.tile_graph` first."
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
    wsi.tables[feature_key].layers[layer_key] = np.asarray(A_spatial)
