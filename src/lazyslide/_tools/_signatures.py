from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from anndata import AnnData


def signatures(
    adata: AnnData,
    groupby: str,
    method: str = "t-test",
):
    """
    Find signature features for a given condition.
    """
    import scanpy as sc

    sc.tl.rank_genes_groups(adata, groupby=groupby, method=method)


def score(
    adata: AnnData,
    group: str,
    method: str = "dot_product",
    n_features: int | str = 100,
):
    """
    Score cells based on a signature.
    """
    import scanpy as sc

    group_df = sc.get.rank_genes_groups_df(adata, group=group)
    # Get the top n_features and low n_features
    weights = np.zeros(adata.n_vars)

    if n_features == "all":
        weights[group_df["names"].astype(int)] = group_df["scores"]
    else:
        top = group_df.head(n_features)
        low = group_df.tail(n_features)
        weights[top["names"].astype(int)] = top["scores"]
        weights[low["names"].astype(int)] = low["scores"]

    # Score cells
    if method == "dot_product":
        scores = adata.X @ weights
    else:
        raise NotImplementedError(f"Method {method} is not implemented.")
    adata.obs[f"{group}_score"] = scores


def associate(
    associate_matrix: AnnData,
    score_key: str = None,
    var_key: str = None,
    method: Literal[
        "pearson", "spearman", "kendall", "linear_reg", "logreg"
    ] = "pearson",
    key_added: str = "correlation",
):
    """
    Associate scores with other omics.
    """
    associate_df = pd.DataFrame(associate_matrix.X, index=associate_matrix.obs.index)
    scores = associate_matrix.obs[score_key]

    if method in {"pearson", "spearman", "kendall"}:
        corr = associate_df.corrwith(scores, method=method)
    elif method == "linear_reg":
        from scipy.stats import linregress

        corr = associate_df.apply(lambda x: linregress(x, scores).rvalue)
    elif method == "logreg":
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression()
        clf.fit(associate_df, scores)
        corr = clf.coef_
    else:
        raise NotImplementedError(f"Method {method} is not implemented.")

    associate_matrix.var[key_added] = np.asarray(corr)
