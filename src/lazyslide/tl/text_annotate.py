from typing import List, Literal

import numpy as np
import pandas as pd

from lazyslide._const import Key
from wsi_data import WSIData


def text_embedding(
    texts: List[str],
    method: Literal["plip", "conch"] = "plip",
):
    """Embed the text

    Parameters
    ----------
    texts : List[str]
        The list of texts.
    model : Literal["plip", "conch"], default: "plip"
        The text embedding model.
    tile_key : str, default: "tiles"
        The tile key.
    """
    import torch

    if method == "plip":
        from lazyslide.models import PLIP

        model = PLIP()
    elif method == "conch":
        from lazyslide.models import CONCH

        model = CONCH()
    else:
        raise ValueError(f"Invalid method: {method}")

    # use numpy record array to store the embeddings
    with torch.inference_mode():
        embeddings = model.encode_text(texts).detach().cpu().numpy()
    return pd.DataFrame(embeddings, index=texts)


def text_annotate(
    wsi: WSIData,
    texts: pd.DataFrame,
    method: Literal["plip", "conch"] = "plip",
    tile_key: str = Key.tiles,
    feature_key: str = None,
):
    if feature_key is None:
        feature_key = method
    feature_key = wsi._check_feature_key(feature_key, tile_key)

    feature_X = wsi.sdata.labels[feature_key].values
    similarity_score = np.dot(texts.values, feature_X.T)
    return similarity_score
