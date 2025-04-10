import numpy as np
from wsidata import WSIData
from wsidata.io import update_shapes_data, add_shapes

from lazyslide._const import Key


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
            "Please install scanpy to use this function, try `pip install scanpy`."
        )
    feature_key = wsi._check_feature_key(feature_key, tile_key)
    adata = wsi.fetch.features_anndata(feature_key, tile_key, tile_graph=False)
    sc.pp.scale(adata, layer=layer)
    sc.pp.pca(adata, layer=layer)
    sc.pp.neighbors(adata)
    sc.tl.leiden(adata, flavor="igraph", key_added=key_added, resolution=resolution)
    # Add to tile table
    update_shapes_data(wsi, tile_key, {key_added: adata.obs[key_added].to_numpy()})


def tile_shaper(
    wsi: WSIData,
    groupby: str = "domain",
    tile_key: str = Key.tiles,
    key_added: str = "domain_shapes",
):
    import geopandas as gpd
    from lazyslide.cv import BinaryMask
    from shapely.affinity import scale, translate

    result = []

    tile_table = wsi[tile_key]

    spec = wsi.tile_spec(tile_key)

    # To avoid large memory allocation of mask, get domain in each tissue
    for _, tissue_group in tile_table.groupby("tissue_id"):
        for name, group in tissue_group.groupby(groupby):
            bounds = (group.bounds / spec.base_height).astype(int)
            minx, miny, maxx, maxy = (
                bounds["minx"].min(),
                bounds["miny"].min(),
                bounds["maxx"].max(),
                bounds["maxy"].max(),
            )
            w, h = int(maxx - minx), int(maxy - miny)
            mask = np.zeros((h, w), dtype=np.uint8)
            for _, row in bounds.iterrows():
                mask[row["miny"] - miny, row["minx"] - minx] = 1
            polys = BinaryMask(mask).to_polygons()
            # scale back
            polys = [
                scale(
                    poly, xfact=spec.base_height, yfact=spec.base_height, origin=(0, 0)
                )
                for poly in polys
            ]
            # translate
            polys = [
                translate(
                    poly, xoff=minx * spec.base_height, yoff=miny * spec.base_height
                )
                for poly in polys
            ]
            for poly in polys:
                result.append([name, poly])

    domain_shapes = gpd.GeoDataFrame(data=result, columns=[groupby, "geometry"])
    add_shapes(wsi, key_added, domain_shapes)
    # return domain_shapes
