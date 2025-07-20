from typing import List, Literal

import numpy as np
import pandas as pd
from wsidata import WSIData
from wsidata.io import add_features

from lazyslide._const import Key
from lazyslide._utils import get_torch_device


def text_embedding(
    texts: List[str],
    model: Literal["plip", "conch", "omiclip"] = "plip",
    device: str = None,
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
    device : str, optional
        The device to use for computation (e.g., 'cpu', 'cuda', 'mps').
        If None, will use CUDA if available, otherwise CPU.

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

    # Determine device
    if device is None:
        device = get_torch_device()

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

    # Move model to the specified device
    model_ins.to(device)

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
    softmax=False,
    scoring_func: callable = None,
):
    """
    Compute the similarity between text and image.

    .. note::
        Prerequisites:

        - The image features should be extracted using
          :func:`zs.tl.feature_extraction <lazyslide.tl.feature_extraction>`.
        - The text embeddings should be computed using
          :func:`zs.tl.text_embedding <lazyslide.tl.text_embedding>`.

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
        The key to store the similarity scores. If None, defaults to
        '{feature_key}_text_similarity'.
    softmax : bool, default: False
        Whether to apply softmax to the similarity scores.
    distance_metric : str or callable, optional
        The distance metric from scipy.spatial.distance to use instead of
        dot product. Can be a string metric name or a callable function.
        If provided, distances will be computed and converted to similarities
        (1 - distance). Common string options include 'cosine', 'euclidean',
        'manhattan', 'chebyshev', etc. If None, uses dot product similarity.
        Cannot be used together with scoring_func.
    scoring_func : callable, optional
        A custom scoring/similarity function that takes two matrices and
        returns a similarity score matrix (higher = more similar). Should
        have same signature as np.dot: func(X, Y) where X is (n_texts,
        feature_dim) and Y is (feature_dim, n_features), returning
        (n_texts, n_features). If provided, this takes precedence over
        distance_metric and dot product. Cannot be used together with
        distance_metric.

    Returns
    -------
    None

    .. note::
        The similarity scores will be saved in the :bdg-danger:`tables`
        slot of the spatial data object.

    Examples
    --------

    .. code-block:: python

        >>> import lazyslide as zs
        >>> # Using dot product similarity (default)
        >>> zs.tl.text_image_similarity(wsi, embeddings, model="plip",
        ...                             tile_key="text_tiles",
        ...                             softmax=True)
        >>> # Using scipy distance functions
        >>> zs.tl.text_image_similarity(wsi, embeddings, model="plip",
        ...                             tile_key="text_tiles",
        ...                             distance_metric="euclidean")
        >>> # Using custom scoring function
        >>> zs.tl.text_image_similarity(wsi, embeddings, model="plip",
        ...                             tile_key="text_tiles",
        ...                             scoring_func=custom_scoring_func)
    """

    if feature_key is None:
        feature_key = model
    feature_key = wsi._check_feature_key(feature_key, tile_key)
    key_added = key_added or f"{feature_key}_text_similarity"

    feature_X = wsi.tables[feature_key].X

    if scoring_func is not None:
        if callable(scoring_func):
            try:
                similarity_score = scoring_func(text_embeddings.values, feature_X.T).T
            except Exception as e:
                raise ValueError(
                    f"Error in custom scoring_func: {str(e)}. "
                    f"Function should accept (n_texts, feature_dim) and "
                    f"(feature_dim, n_features) matrices and return "
                    f"(n_texts, n_features) similarity matrix."
                ) from e
        elif isinstance(scoring_func, str):
            from scipy.spatial.distance import cdist

            similarity_score = cdist(
                text_embeddings.values, feature_X, metric=scoring_func
            ).T
    else:
        similarity_score = np.dot(text_embeddings.values, feature_X.T).T

    if softmax:
        from scipy.special import softmax

        similarity_score = softmax(similarity_score, axis=1)

    add_features(
        wsi,
        key_added,
        tile_key,
        similarity_score,
        var=pd.DataFrame(index=text_embeddings.index),
    )
