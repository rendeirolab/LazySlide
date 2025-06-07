from __future__ import annotations

import warnings
from itertools import chain

import numpy as np
import pandas as pd
from anndata import AnnData
from numba import njit
from scipy.sparse import SparseEfficiencyWarning, csr_matrix, isspmatrix_csr, spmatrix
from scipy.spatial import Delaunay
from wsidata import WSIData
from wsidata.io import add_table

from lazyslide._const import Key


def tile_graph(
    wsi: WSIData,
    n_neighs: int = 6,
    n_rings: int = 1,
    delaunay=False,
    transform: str = None,
    set_diag: bool = False,
    tile_key: str = Key.tiles,
    table_key: str = None,
):
    """
    Compute the spatial graph of the tiles.

    Parameters
    ----------
    wsi : :class:`WSIData <wsidata.WSIData>`
        The WSIData object to work on.
    n_neighs : int, default: 6
        The number of neighbors to consider.
    n_rings : int, default: 1
        The number of rings to consider.
    delaunay : bool, default: False
        Whether to use Delaunay triangulation.
    transform : str, default: None
        The transformation to apply to the graph.
    set_diag : bool, default: False
        Whether to set the diagonal to 1.
    tile_key : str, default: 'tiles'
        The tile key.
    table_key : str, default: None
        The table key to store the graph.

    Returns
    -------
    :class:`AnnData <anndata.AnnData>`
        The tiles with spatial connectivities and distances in an anndata format. |
        Added to :code:`tile_graph | {key_added}` in :bdg-danger:`tables` slot of the WSIData object.

    Examples
    --------
    .. code-block:: python

        >>> import lazyslide as zs
        >>> wsi = zs.datasets.sample()
        >>> zs.pp.find_tissues(wsi)
        >>> zs.pp.tile_graph(wsi)
        >>> wsi['tile_graph']


    """
    coords = wsi[tile_key].bounds[["minx", "miny"]].values
    Adj, Dst = _spatial_neighbor(
        coords, n_neighs, delaunay, n_rings, transform, set_diag
    )

    conns_key = "spatial_connectivities"
    dists_key = "spatial_distances"
    neighbors_dict = {
        "connectivities_key": conns_key,
        "distances_key": dists_key,
        "params": {
            "n_neighbors": n_neighs,
            "transform": transform,
        },
    }
    # TODO: Store in a anndata object
    if table_key is None:
        table_key = Key.tile_graph(tile_key)
    if table_key not in wsi:
        table = AnnData(
            obs=pd.DataFrame(index=np.arange(coords.shape[0], dtype=int).astype(str)),
            obsp={conns_key: Adj, dists_key: Dst},
            uns={"spatial": neighbors_dict},
        )
        add_table(wsi, table_key, table)
    else:
        table = wsi[table_key]
        table.obsp[conns_key] = Adj
        table.obsp[dists_key] = Dst
        table.uns["spatial"] = neighbors_dict


def _spatial_neighbor(
    coords,
    n_neighs: int = 6,
    delaunay: bool = False,
    n_rings: int = 1,
    transform: str = None,
    set_diag: bool = False,
) -> tuple[csr_matrix, csr_matrix]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SparseEfficiencyWarning)
        Adj, Dst = _build_grid(
            coords,
            n_neighs=n_neighs,
            n_rings=n_rings,
            delaunay=delaunay,
            set_diag=set_diag,
        )

    Adj.eliminate_zeros()
    Dst.eliminate_zeros()

    # check transform
    if transform == "spectral":
        Adj = _transform_a_spectral(Adj)
    elif transform == "cosine":
        Adj = _transform_a_cosine(Adj)
    elif transform == "none" or transform is None:
        pass
    else:
        raise NotImplementedError(f"Transform `{transform}` is not yet implemented.")

    return Adj, Dst


def _build_grid(
    coords,
    n_neighs: int,
    n_rings: int,
    delaunay: bool = False,
    set_diag: bool = False,
) -> tuple[csr_matrix, csr_matrix]:
    if n_rings > 1:
        Adj: csr_matrix = _build_connectivity(
            coords,
            n_neighs=n_neighs,
            neigh_correct=True,
            set_diag=True,
            delaunay=delaunay,
            return_distance=False,
        )
        Res, Walk = Adj, Adj
        for i in range(n_rings - 1):
            Walk = Walk @ Adj
            Walk[Res.nonzero()] = 0.0
            Walk.eliminate_zeros()
            Walk.data[:] = i + 2.0
            Res = Res + Walk
        Adj = Res
        Adj.setdiag(float(set_diag))
        Adj.eliminate_zeros()

        Dst = Adj.copy()
        Adj.data[:] = 1.0
    else:
        Adj = _build_connectivity(
            coords,
            n_neighs=n_neighs,
            neigh_correct=True,
            delaunay=delaunay,
            set_diag=set_diag,
        )
        Dst = Adj.copy()

    Dst.setdiag(0.0)

    return Adj, Dst


def _build_connectivity(
    coords,
    n_neighs: int,
    radius: float | tuple[float, float] | None = None,
    delaunay: bool = False,
    neigh_correct: bool = False,
    set_diag: bool = False,
    return_distance: bool = False,
) -> csr_matrix | tuple[csr_matrix, csr_matrix]:
    from sklearn.metrics import euclidean_distances
    from sklearn.neighbors import NearestNeighbors

    N = coords.shape[0]
    if delaunay:
        tri = Delaunay(coords)
        indptr, indices = tri.vertex_neighbor_vertices
        Adj = csr_matrix(
            (np.ones_like(indices, dtype=np.float64), indices, indptr), shape=(N, N)
        )

        if return_distance:
            # fmt: off
            dists = np.array(list(chain(*(
                euclidean_distances(coords[indices[indptr[i]: indptr[i + 1]], :], coords[np.newaxis, i, :])
                for i in range(N)
                if len(indices[indptr[i]: indptr[i + 1]])
            )))).squeeze()
            Dst = csr_matrix((dists, indices, indptr), shape=(N, N))
            # fmt: on
    else:
        r = (
            1
            if radius is None
            else radius
            if isinstance(radius, (int, float))
            else max(radius)
        )
        tree = NearestNeighbors(n_neighbors=n_neighs, radius=r, metric="euclidean")
        tree.fit(coords)

        if radius is None:
            dists, col_indices = tree.kneighbors()
            dists, col_indices = dists.reshape(-1), col_indices.reshape(-1)
            row_indices = np.repeat(np.arange(N), n_neighs)
            if neigh_correct:
                dist_cutoff = np.median(dists) * 1.3  # there's a small amount of sway
                mask = dists < dist_cutoff
                row_indices, col_indices, dists = (
                    row_indices[mask],
                    col_indices[mask],
                    dists[mask],
                )
        else:
            dists, col_indices = tree.radius_neighbors()
            row_indices = np.repeat(np.arange(N), [len(x) for x in col_indices])
            dists = np.concatenate(dists)
            col_indices = np.concatenate(col_indices)

        Adj = csr_matrix(
            (np.ones_like(row_indices, dtype=np.float64), (row_indices, col_indices)),
            shape=(N, N),
        )
        if return_distance:
            Dst = csr_matrix((dists, (row_indices, col_indices)), shape=(N, N))

    # radius-based filtering needs same indices/indptr: do not remove 0s
    Adj.setdiag(1.0 if set_diag else Adj.diagonal())
    if return_distance:
        Dst.setdiag(0.0)
        return Adj, Dst

    return Adj


@njit
def outer(indices, indptr, degrees):
    res = np.empty_like(indices, dtype=np.float64)
    start = 0
    for i in range(len(indptr) - 1):
        ixs = indices[indptr[i] : indptr[i + 1]]
        res[start : start + len(ixs)] = degrees[i] * degrees[ixs]
        start += len(ixs)

    return res


def _transform_a_spectral(a: spmatrix) -> spmatrix:
    if not isspmatrix_csr(a):
        a = a.tocsr()
    if not a.nnz:
        return a

    degrees = np.squeeze(np.array(np.sqrt(1.0 / a.sum(axis=0))))
    a = a.multiply(outer(a.indices, a.indptr, degrees))
    a.eliminate_zeros()

    return a


def _transform_a_cosine(a: spmatrix) -> spmatrix:
    from sklearn.metrics.pairwise import cosine_similarity

    return cosine_similarity(a, dense_output=False)
