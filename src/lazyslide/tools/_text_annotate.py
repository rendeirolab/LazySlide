from typing import List, Literal

import numpy as np
import pandas as pd

from lazyslide._const import Key
from wsidata import WSIData


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
    tile_key : str, default: "tiles"
        The tile key.
    """
    import torch

    if model == "plip":
        from lazyslide.models import PLIP

        model_ins = PLIP()
    elif model == "conch":
        from lazyslide.models import CONCH

        model_ins = CONCH()
    else:
        raise ValueError(f"Invalid model: {model}")

    # use numpy record array to store the embeddings
    with torch.inference_mode():
        embeddings = model_ins.encode_text(texts).detach().cpu().numpy()
    return pd.DataFrame(embeddings, index=texts)


def text_annotate(
    wsi: WSIData,
    text_embeddings: pd.DataFrame,
    model: Literal["plip", "conch"] = "plip",
    tile_key: str = Key.tiles,
    feature_key: str = None,
):
    if feature_key is None:
        feature_key = model
    feature_key = wsi._check_feature_key(feature_key, tile_key)

    feature_X = wsi.sdata.tables[feature_key].X
    similarity_score = np.dot(text_embeddings.values, feature_X.T)
    wsi.update_shapes_data(
        tile_key,
        {text: similarity_score[i] for i, text in enumerate(text_embeddings.index)},
    )
