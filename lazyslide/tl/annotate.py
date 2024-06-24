from typing import Literal, List

import pandas as pd

from lazyslide.wsi import WSI


def annotate(
    wsi: WSI,
    texts: List[str],
    method: Literal["plip", "conch"] = "plip",
    tile_key="tiles",
    domain_feature_key=None,
    device="cpu",
    key_added="annotation",
):
    """Annotate the WSI features with text

    Parameters
    ----------
    wsi : WSI
        The whole slide image object.
    texts : List[str] | Dict[str, np.ndarray]
        The list of texts to annotate or pre-computed embeddings features.
    method : Literal["plip", "conch"], default: "plip"
        The annotation method.
    tile_key : str, default: "tiles"
        The tile key.
    feature_key : str, default: None
        The feature key.

    """
    import anndata as ad
    import scanpy as sc

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

    if isinstance(texts, list):
        model.to(device)
        texts_embeddings = model.encode_text(texts).detach().cpu().numpy()

    adata = wsi.sdata.tables[f"{tile_key}/{feature_key}"]
    domain_adata = wsi.sdata.tables[f"{tile_key}/{domain_feature_key}"]
    sim_scores = adata.X @ texts_embeddings.T
    score_adata = ad.AnnData(
        X=sim_scores, obs=domain_adata.obs, var=pd.DataFrame(index=texts)
    )
    sc.tl.rank_genes_groups(score_adata, groupby="domain", method="t-test")
    names = score_adata.uns["rank_genes_groups"]["names"]
    mapper = dict(zip(names.dtype.names, names[0]))
    domain_adata.obs[key_added] = domain_adata.obs["domain"].map(mapper)
    return score_adata
