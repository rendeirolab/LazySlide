import numpy as np
from wsidata import WSIData
from wsidata.io import add_shapes, update_shapes_data

from lazyslide._const import Key


def spatial_domain(
    wsi: WSIData,
    feature_key: str,
    tile_key: str = Key.tiles,
    layer: str = None,
    resolution: float = 0.1,
    key_added: str = "domain",
):
    """
    Perform unsupervised spatial domain segmentation on a WSI using feature embeddings.

    This function applies scaling, PCA, neighborhood graph construction, and Leiden clustering
    to identify spatial domains within the WSI based on the provided features.

    Parameters
    ----------
    wsi : :class:`WSIData <wsidata.WSIData>`
        The whole-slide image object.
    feature_key : str
        The key for the feature table to use.
    tile_key : str, default: "tiles"
        The key for the tile table.
    layer : str, optional
        The layer in the feature table to use for clustering.
    resolution : float, optional
        The resolution parameter for Leiden clustering. Defaults to 0.1.
    key_added : str, optional
        The key under which to store the domain labels. Defaults to "domain".

    Returns
    -------
    None
        The domain labels are added to the tile table in the WSIData object.

    Examples
    --------

    .. code-block:: python

        >>> import lazyslide as zs
        >>> wsi = zs.datasets.sample()
        >>> zs.pp.find_tissues(wsi)
        >>> zs.pp.tile_tissues(wsi, 256, mpp=0.5)
        >>> zs.tl.feature_extraction(wsi, "resnet50")
        >>> zs.pp.tile_graph(wsi)
        >>> zs.tl.spatial_features(wsi)
        >>> zs.tl.spatial_domain(wsi, layer="spatial_features", feature_key="resnet", resolution=0.3)

    """
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
    """
    Return the domain shapes of the WSI by merging tiles with the same types
    that are spatially aggregated into polygons using geopandas dissolve.

    Parameters
    ----------
    wsi : :class:`WSIData <wsidata.WSIData>`
        The WSIData object.
    groupby : str
        The groupby key.
    tile_key : str
        The tile key.
    key_added : str
        The key to add the shapes to.

    Returns
    -------
    :class:`GeoDataFrame <geopandas.GeoDataFrame>`
        Added to to the :bdg-danger:`shapes` slot of the WSIData object.

    Examples
    --------

    .. code-block:: python

        >>> import lazyslide as zs
        >>> wsi = zs.datasets.sample()
        >>> zs.tl.tile_shaper(wsi, groupby="domain")

    """
    import geopandas as gpd
    from shapely.geometry import box

    tile_table = wsi[tile_key]

    # Create box geometries from tile bounds
    geometries = []
    for _, row in tile_table.iterrows():
        geom = box(
            row.bounds["minx"],
            row.bounds["miny"],
            row.bounds["maxx"],
            row.bounds["maxy"],
        )
        geometries.append(geom)

    # Create GeoDataFrame with tile geometries and groupby column
    tiles_gdf = gpd.GeoDataFrame(
        {
            groupby: tile_table[groupby],
            "tissue_id": tile_table["tissue_id"],
            "geometry": geometries,
        }
    )

    # Group by tissue_id and groupby column, then dissolve to merge adjacent tiles
    domain_shapes = (
        tiles_gdf.dissolve(by=[groupby, "tissue_id"]).explode().reset_index(drop=True)
    )

    # Keep only the groupby column and geometry
    domain_shapes = domain_shapes[[groupby, "geometry"]]

    add_shapes(wsi, key_added, domain_shapes)
    # return domain_shapes
