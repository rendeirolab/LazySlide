from typing import Literal

from lazyslide._const import Key
from wsi_data import WSIData


def spatial_domain(
    wsi: WSIData,
    feature_key: str,
    tile_key: str = Key.tiles,
    method: Literal["leiden", "utag"] = "leiden",
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
    if method == "utag" and not feature_key.endswith("_utag"):
        feature_key = f"{feature_key}_utag"
    adata = wsi.get.features_anndata(feature_key, tile_key, tile_graph=False)
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.leiden(adata, flavor="igraph", key_added=key_added, resolution=resolution)
    # Add to tile table
    wsi.update_shapes_data(tile_key, {key_added: adata.obs[key_added].to_numpy()})
