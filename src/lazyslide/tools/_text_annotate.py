from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from wsidata import WSIData
from wsidata.io import add_features

from lazyslide import _api
from lazyslide._const import Key

if TYPE_CHECKING:
    import torch
    from lazyslide_models import ImageTextModelProtocol


def text_embedding(
    texts: list[str],
    model: str | ImageTextModelProtocol = "plip",
    amp: bool | None = None,
    autocast_dtype: torch.dtype = None,
    device: str = "cpu",
    *,
    compile: bool | None = None,
    compile_kws: dict | None = None,
):
    """Embed the text into a vector in the text-vision co-embedding of a
    :term:`multimodal model`, for example
    `PLIP <https://www.nature.com/articles/s41591-023-02504-3>`_ (the default),
    `CONCH <https://www.nature.com/articles/s41591-024-02856-4>`_ or
    `OmiCLIP <https://www.nature.com/articles/s41592-025-02707-1>`_.

    Parameters
    ----------
    texts : List[str]
        The list of texts.
    model : str or ImageTextModelProtocol, default: "plip"
        The text embedding :term:`multimodal model`: a model registry key (see
        :ref:`models-section`) or a model instance.
    amp : bool, optional
        Whether to use automatic mixed precision (AMP) for inference.
    autocast_dtype : torch.dtype, optional
        The dtype for automatic mixed precision.
    device : str, default: "cpu"
        The device to use for computation (e.g., 'cpu', 'cuda', 'mps').
        Defaults to CPU on purpose: embedding a handful of short strings is a
        tiny amount of compute, and moving the text encoder to a GPU/MPS device
        costs more than it saves. This one does not follow
        :code:`settings.device` — pass the device explicitly to override.
    compile : bool, optional
        Whether to compile the model with :func:`torch.compile`.
        Compilation is best-effort and is silently skipped for models
        that do not support it.
    compile_kws : dict, optional
        Keyword arguments passed to :func:`torch.compile`.

    Returns
    -------
    :class:`DataFrame <pandas.DataFrame>`
        The :term:`embeddings <embedding>` of the texts, with texts as index.

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
    from lazyslide_models import MODEL_REGISTRY

    # NOT settings.device: text embedding is small enough that a device
    # transfer costs more than the compute. See the ``device`` docstring.

    if isinstance(model, str):
        model = MODEL_REGISTRY[model]()
    model.to(device)
    model = _api.maybe_compile(model, compile, compile_kws)

    amp_ctx = _api.autocast(device, amp, autocast_dtype)
    with amp_ctx, torch.inference_mode():
        # use numpy record array to store the embeddings
        embeddings = model.encode_text(texts).detach().cpu().numpy()
    return pd.DataFrame(embeddings, index=texts)


def text_image_similarity(
    wsi: WSIData,
    text_embeddings: pd.DataFrame,
    model: str = "plip",
    tile_key: str = Key.tiles,
    feature_key: str | None = None,
    key_added: str | None = None,
    normalize: bool = True,
    softmax=False,
    scoring_func: Callable | None = None,
):
    """
    Compute the similarity between text and image.

    .. note::
        Prerequisites:

        - The image :term:`features` should be extracted using
          :func:`zs.tl.feature_extraction <lazyslide.tl.feature_extraction>`.
        - The text :term:`embeddings <embedding>` should be computed using
          :func:`zs.tl.text_embedding <lazyslide.tl.text_embedding>`.

    Parameters
    ----------
    wsi : :class:`WSIData <wsidata.WSIData>`
        The WSIData object to work on.
    text_embeddings : :class:`DataFrame <pandas.DataFrame>`
        The embeddings of the texts, with texts as index.
    model : str, default: "plip"
        The name of the model the image features were extracted with. Only used
        to find them when ``feature_key`` is None.
    tile_key : str, default: 'tiles'
        The tile key.
    feature_key : str, default: None
        The feature key.
    key_added : str, default: None
        The key to store the similarity scores. If None, defaults to
        '{feature_key}_text_similarity'.
    normalize : bool, default: True
        Apply L2 normalization to the :term:`tile` :term:`features` before computing the
        similarity score to the text embeddings.
    softmax : bool, default: False
        Whether to apply softmax to the similarity scores.
    scoring_func : callable, optional
        A custom scoring/similarity function that takes two matrices and
        returns a similarity score matrix (higher = more similar). Should
        have same signature as np.dot: func(X, Y) where X is (n_texts,
        feature_dim) and Y is (feature_dim, n_features), returning
        (n_texts, n_features).

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
    if normalize:
        # Use default parameters from torch.nn.functional.normalize
        eps = 1e-12
        norm = np.linalg.norm(feature_X, ord=2, axis=1, keepdims=True)
        feature_X = feature_X / np.maximum(norm, eps)

    if scoring_func is not None:
        if callable(scoring_func):
            try:
                similarity_score = scoring_func(text_embeddings.values, feature_X.T).T
            except Exception as e:
                raise ValueError(
                    f"Error in custom scoring_func: {e!s}. "
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
