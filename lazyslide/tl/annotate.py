import warnings
from typing import Literal, List

import numpy as np
import pandas as pd

from lazyslide.wsi import WSI


def annotate(
    wsi: WSI,
    texts: List[str] | "AnnData",  # noqa: F821
    method: Literal["plip", "conch"] = "plip",
    model=None,
    tile_key="tiles",
    domain_key="domain",
    device="cpu",
    key_added="annotation",
):
    """Annotate the WSI features with text

    Parameters
    ----------
    wsi : WSI
        The whole slide image object.
    texts : List[str] | Dict[str, np.ndarray] | AnnData
        The list of texts to annotate or
        pre-computed embeddings features in a dictionary or AnnData.
    method : Literal["plip", "conch"], default: "plip"
        The annotation method.
    tile_key : str, default: "tiles"
        The tile key.
    feature_key : str, default: None
        The feature key.

    """
    import anndata as ad
    import scanpy as sc
    import torch

    if model is None:
        if method == "plip":
            from lazyslide.models import PLIP

            model = PLIP()
            feature_key = "PLIP"
        elif method == "conch":
            from lazyslide.models import CONCH

            model = CONCH()
            feature_key = "CONCH"
        else:
            raise ValueError(f"Invalid method: {method}")
    else:
        feature_key = model.__class__.__name__

    if f"{tile_key}_{feature_key}" not in wsi.sdata.tables:
        raise ValueError(_run_model_error_msg(feature_key, feature_key))

    model.to(device)
    if isinstance(texts, list):
        with torch.inference_mode():
            texts_embeddings = model.encode_text(texts).detach().cpu().numpy()
        texts_adata = ad.AnnData(X=texts_embeddings, obs=pd.DataFrame(index=texts))
    elif isinstance(texts, ad.AnnData):
        texts_adata = texts
        texts_embeddings = texts.X
        texts = texts.obs.index
    elif isinstance(texts, dict):
        obs_index = []
        X = []
        for text, embedding in texts.items():
            obs_index.append(text)
            X.append(embedding)
        texts_embeddings = np.vstack(X)
        texts_adata = ad.AnnData(X=texts_embeddings, obs=pd.DataFrame(index=obs_index))
        texts = obs_index
    else:
        raise ValueError("Invalid type for texts.")

    if tile_key not in wsi.sdata.points:
        raise ValueError(f"Tile key {tile_key} not found.")
    has_domain = domain_key in wsi.get_tiles_table(tile_key).columns
    if has_domain:
        domains = wsi.get_tiles_table(tile_key)[domain_key].to_numpy()
    else:
        warnings.warn(f"Domain key {domain_key} not found in the table.", stacklevel=2)

    adata = wsi.sdata.tables[f"{tile_key}_{feature_key}"]

    sim_scores = adata.X @ texts_embeddings.T
    score_adata = ad.AnnData(
        X=sim_scores,
        obs=pd.DataFrame({domain_key: domains}) if has_domain else None,
        var=pd.DataFrame(index=texts),
    )
    # TODO: Auto-annotate by getting the terms with highest mean and less variance in the domain
    if has_domain:
        mean_ranks = {}
        for domain, df in score_adata.obs.groupby(
            domain_key, observed=True, sort=False
        ):
            mean_score = score_adata[df.index, :].X.mean(axis=0)
            rank = np.argsort(mean_score)
            mean_ranks[domain] = np.arange(len(rank))[::-1][rank] + 1

        var_ranks = {}
        sc.tl.rank_genes_groups(score_adata, groupby="domain", method="t-test")
        names = pd.DataFrame(score_adata.uns["rank_genes_groups"]["names"]).reset_index(
            names=["rank"]
        )
        for domain in names.columns:
            if domain != "rank":
                var_ranks[domain] = (
                    names.set_index(domain).loc[texts, "rank"] + 1
                ).values

        domain_mapper = {}
        for domain in mean_ranks.keys():
            final_scores = 3 * mean_ranks[domain] + 7 * var_ranks[domain]
            ix = np.argmin(final_scores)
            domain_mapper[domain] = texts[ix]

    # Add the name to domain
    if has_domain:
        wsi.add_tiles_data(
            {key_added: [domain_mapper[domain] for domain in domains]}, tile_key
        )
    # Save the text embeddings
    wsi.sdata.tables["text_embeddings"] = texts_adata
    # Save the similarity scores
    wsi.add_features(
        sim_scores, tile_key, f"{feature_key}_scores", var=pd.DataFrame(index=texts)
    )


def _run_model_error_msg(feature_key: str, model_name: str):
    upper_model_name = model_name.upper()
    lower_model_name = model_name.lower()
    return (
        f"Feature {feature_key} not found. "
        f"Please run the feature extraction method with {feature_key} first.\n"
        f">>> from lazyslide.models import {upper_model_name}\n"
        f">>> {lower_model_name} = {upper_model_name}()\n"
        f">>> zs.tl.feature_extraction(wsi, {lower_model_name})\n"
    )
