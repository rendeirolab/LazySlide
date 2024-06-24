from typing import Literal

import numpy as np

from lazyslide.wsi import WSI, TileSpec


def anatomical_domain(
    wsi: WSI,
    tile_key: str = "tiles",
    feature_key: str = None,
    method: Literal["leiden", "utag"] = "leiden",
    resolution: float = 1.0,
):
    """Return the unsupervised domain of the WSI"""
    try:
        import scanpy as sc
    except ImportError:
        raise ImportError(
            "Please install scanpy to use this function, " "try `pip install scanpy`."
        )

    adata = wsi.sdata.tables[f"{tile_key}/{feature_key}"]
    tile_spec = TileSpec(**wsi.sdata.tables[f"{tile_key}_spec"].uns["tiles_spec"])

    if method == "utag":
        # Calculate UTAG features
        # The radius is the diagonal of the tile
        r = np.sqrt(tile_spec.ops_width**2 + tile_spec.ops_height**2)

        # Compute the spatial connectivity
        from scipy.spatial import KDTree

        tree = KDTree(adata.obsm["spatial"])
        A = tree.sparse_distance_matrix(tree, r).toarray()
        A = A + np.eye(A.shape[0])
        # L1 normalization
        A = np.asarray(A)
        # Calculate the L1 norm for each row
        norms = np.sum(np.abs(A), axis=1, keepdims=True)
        # Normalize the array
        A_norm = A / norms
        A_spatial = np.transpose(adata.X) @ A_norm
        A_spatial = np.transpose(A_spatial)

        adata.layers["UTAG"] = A_spatial
        layer = "UTAG"
    else:
        layer = None

    sc.pp.pca(adata, layer=layer)
    sc.pp.neighbors(adata)
    sc.tl.leiden(adata, flavor="igraph", key_added="domain", resolution=resolution)


def domain_props(
    wsi: WSI,
):
    """Calculate the geometrical properties of anatomical domains of the WSI"""
    pass
