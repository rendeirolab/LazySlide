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
    Integrate spatial tile context with vision features with `UTAG <https://doi.org/10.1038/s41592-022-01657-2>`_.

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

    Returns
    -------
    The transformed feature with UTAG.

    - The transformed feature will be added to :bdg-danger:`tables` slot of the spatial data object.
    - The transformed feature will be stored in the `utag` layer of the feature table.

    Examples
    --------
    .. code-block:: python

        >>> import lazyslide as zs
        >>> wsi = zs.datasets.sample()
        >>> zs.pp.find_tissues(wsi)
        >>> zs.pp.tile_tissues(wsi, 256, mpp=0.5)
        >>> zs.tl.feature_extraction(wsi, "resnet50")
        >>> zs.pp.tile_graph(wsi)
        >>> zs.tl.feature_utag(wsi, "resnet50")
        >>> wsi["resnet50"].layers["utag"]

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
