from typing import Literal

from lazyslide._const import Key
from wsidata import WSIData
from wsidata.io import update_shapes_data


def spatial_domain(
    wsi: WSIData,
    feature_key: str,
    tile_key: str = Key.tiles,
    layer: str = None,
    resolution: float = 0.1,
    key_added: str = "domain",
):
    """Return the unsupervised domain of the WSI"""
    try:
        import scanpy as sc
    except ImportError:
        raise ImportError(
            "Please install scanpy to use this function, " "try `pip install scanpy`."
        )
    feature_key = wsi._check_feature_key(feature_key, tile_key)
    adata = wsi.fetch.features_anndata(feature_key, tile_key, tile_graph=False)
    sc.pp.pca(adata, layer=layer)
    sc.pp.neighbors(adata)
    sc.tl.leiden(adata, flavor="igraph", key_added=key_added, resolution=resolution)
    # Add to tile table
    update_shapes_data(wsi, tile_key, {key_added: adata.obs[key_added].to_numpy()})
