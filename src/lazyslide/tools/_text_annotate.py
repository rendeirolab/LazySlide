from typing import List, Literal

import numpy as np
import pandas as pd
from wsidata import WSIData
from wsidata.io import add_features

from lazyslide._const import Key


def text_embedding(
    texts: List[str],
    model: Literal["plip", "conch"] = "plip",
):
    """Embed the text

    Parameters
    ----------
    texts : List[str]
        The list of texts.
    model : Literal["plip", "conch"], default: "plip"
        The text embedding model.

    Returns
    -------
    pd.DataFrame
        The embeddings of the texts, with texts as index.

    """
    import torch

    if model == "plip":
        from lazyslide.models.multimodal import PLIP

        model_ins = PLIP()
    elif model == "conch":
        from lazyslide.models.multimodal import CONCH

        model_ins = CONCH()
    else:
        raise ValueError(f"Invalid model: {model}")

    # use numpy record array to store the embeddings
    with torch.inference_mode():
        embeddings = model_ins.encode_text(texts).detach().cpu().numpy()
    return pd.DataFrame(embeddings, index=texts)


def text_image_similarity(
    wsi: WSIData,
    text_embeddings: pd.DataFrame,
    model: Literal["plip", "conch"] = "plip",
    tile_key: str = Key.tiles,
    feature_key: str = None,
    key_added: str = None,
):
    """
    Compute the similarity between text and image.

    Parameters
    ----------
    wsi : WSIData
        The WSIData object.
    text_embeddings : pd.DataFrame
        The embeddings of the texts, with texts as index.
        You can use :func:`text_embedding` to get the embeddings.
    model : Literal["plip", "conch"], default: "plip"
        The text embedding model.
    tile_key : str, default: 'tiles'
        The tile key.
    feature_key : str
        The feature key.
    key_added : str

    Returns
    -------

    """

    if feature_key is None:
        feature_key = model
    feature_key = wsi._check_feature_key(feature_key, tile_key)
    key_added = f"{feature_key}_text_similarity" or key_added

    feature_X = wsi.tables[feature_key].X
    similarity_score = np.dot(text_embeddings.values, feature_X.T).T

    add_features(
        wsi,
        key_added,
        tile_key,
        similarity_score,
        var=pd.DataFrame(index=text_embeddings.index),
    )
