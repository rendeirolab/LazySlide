from typing import List, Literal

import numpy as np
import pandas as pd
from wsidata import WSIData
from wsidata.io import add_features

from lazyslide._const import Key


def text_embedding(
    texts: List[str],
    model: Literal["plip", "conch", "omiclip"] = "plip",
):
    """Embed the text into a vector in the text-vision co-embedding using

    - `PLIP <https://www.nature.com/articles/s41591-023-02504-3>`_
    - `CONCH <https://www.nature.com/articles/s41591-024-02856-4>`_
    - `OmiCLIP <https://www.nature.com/articles/s41592-025-02707-1>`_

    Parameters
    ----------
    texts : List[str]
        The list of texts.
    model : Literal["plip", "conch", "omiclip"], default: "plip"
        The text embedding model

    Returns
    -------
    :class:`DataFrame <pandas.DataFrame>`
        The embeddings of the texts, with texts as index.

    Examples
    --------

    .. code-block:: python

        >>> import lazyslide as zs
        >>> wsi = zs.datasets.sample()
        >>> zs.pp.find_tissues(wsi)
        >>> zs.pp.tile_tissues(wsi, 256, mpp=0.5, key_added="text_tiles")
        >>> zs.tl.feature_extraction(wsi, "plip", tile_key="text_tiles")
        >>> terms = ["mucosa", "submucosa", "musclaris", "lymphocyte"]
        >>> zs.tl.text_embedding(terms, model="plip")

    """
    import torch

    if model == "plip":
        from lazyslide.models.multimodal import PLIP

        model_ins = PLIP()
    elif model == "conch":
        from lazyslide.models.multimodal import CONCH

        model_ins = CONCH()
    elif model == "omiclip":
        from lazyslide.models.multimodal import OmiCLIP

        model_ins = OmiCLIP()
    else:
        raise ValueError(f"Invalid model: {model}")

    # use numpy record array to store the embeddings
    with torch.inference_mode():
        embeddings = model_ins.encode_text(texts).detach().cpu().numpy()
    return pd.DataFrame(embeddings, index=texts)


def text_image_similarity(
    wsi: WSIData,
    text_embeddings: pd.DataFrame,
    model: Literal["plip", "conch", "omiclip"] = "plip",
    tile_key: str = Key.tiles,
    feature_key: str = None,
    key_added: str = None,
):
    """
    Compute the similarity between text and image.

    .. note::
        Prerequisites:

        - The image features should be extracted using :func:`zs.tl.feature_extraction <lazyslide.tl.feature_extraction>`.
        - The text embeddings should be computed using :func:`zs.tl.text_embedding <lazyslide.tl.text_embedding>`.

    Parameters
    ----------
    wsi : :class:`WSIData <wsidata.WSIData>`
        The WSIData object to work on.
    text_embeddings : pd.DataFrame
        The embeddings of the texts, with texts as index.
    model : {"plip", "conch", "omiclip"}, default: "plip"
        The text embedding model.
    tile_key : str, default: 'tiles'
        The tile key.
    feature_key : str
        The feature key.
    key_added : str
        The key to store the similarity scores. If None, defaults to '{feature_key}_text_similarity'.

    Returns
    -------
    None

    .. note::
        The similarity scores will be saved as an  to :bdg-danger:`tables` slot of the spatial data object.

    Examples
    --------

    .. code-block:: python

        >>> import lazyslide as zs
        >>> wsi = zs.datasets.sample()
        >>> zs.pp.find_tissues(wsi)
        >>> zs.pp.tile_tissues(wsi, 256, mpp=0.5, key_added="text_tiles")
        >>> zs.tl.feature_extraction(wsi, "plip", tile_key="text_tiles")
        >>> terms = ["mucosa", "submucosa", "musclaris", "lymphocyte"]
        >>> embeddings = zs.tl.text_embedding(terms, model="plip")
        >>> zs.tl.text_image_similarity(wsi, embeddings, model="plip", tile_key="text_tiles")

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
