from __future__ import annotations

from typing import Sequence, List, Iterable

import numpy as np
import pandas as pd
import torch
from wsidata import WSIData

from lazyslide._utils import get_torch_device


def _preprocess_prompts(prompts: List[str | List[str]]) -> List[List[str]]:
    """
    Preprocess the prompts to ensure they are in the correct format.
    """
    processed_prompts = []
    for prompt in prompts:
        if isinstance(prompt, str):
            processed_prompts.append([prompt])
        elif isinstance(prompt, Iterable):
            processed_prompts.append(list(prompt))
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")
    return processed_prompts


def _get_agg_info(
    wsi: WSIData,
    feature_key,
    agg_key: str = None,
    agg_by: str | Sequence[str] = None,
):
    if agg_key is None:
        if agg_by is None:
            agg_key = "agg_slide"
        else:
            if isinstance(agg_by, str):
                agg_by = [agg_by]
            agg_key = f"agg_{'_'.join(agg_by)}"
    agg_info = wsi[feature_key].uns["agg_ops"][agg_key]
    annos = None
    if "keys" in agg_info:
        annos = pd.DataFrame(
            data=agg_info["values"],
            columns=agg_info["keys"],
        )
    return agg_info, annos


def zero_shot_score(
    wsi: WSIData,
    prompts: list[list[str]],
    feature_key,
    *,
    agg_key: str = None,
    agg_by: str | Sequence[str] = None,
    model: str = "prism",
    device: str = None,
):
    """
    Perform zero-shot classification on the WSI

    Supported models:
    - prism: Prism model
    - titan: Titan model

    Corresponding slide-level features are required for the model.


    Parameters
    ----------
    wsi : :class:`wsidata.WSIData`
        The WSI data object.
    prompts : array of str
        The text labels to classify. You can use a list of strings to
        add more information to one class.
    feature_key : str
        The tile features to be used.
    agg_key : str
        The aggregation key
    agg_by : str or list of str
        The aggregation keys that were used to create the slide features.
    model: {"prism", "titan"}
        The model to use for zero-shot classification.
    device : str
        The device to use for inference. If None, the default device will be used.

    """
    if device is None:
        device = get_torch_device()

    prompts = _preprocess_prompts(prompts)

    if model == "prism":
        from lazyslide.models.multimodal import Prism

        model = Prism()
    elif model == "titan":
        from lazyslide.models.multimodal import Titan

        model = Titan()
    model.to(device)
    # Get the embeddings from the WSI
    agg_info, annos = _get_agg_info(
        wsi,
        feature_key,
        agg_key=agg_key,
        agg_by=agg_by,
    )

    all_probs = []
    for ix, f in enumerate(agg_info["features"]):
        f = torch.tensor(f).unsqueeze(0).to(device)
        probs = model.score(f, prompts=prompts)
        all_probs.append(probs)

    all_probs = np.vstack(all_probs)

    named_prompts = [", ".join(p) for p in prompts]
    results = pd.DataFrame(
        data=all_probs,
        columns=named_prompts,
    )
    if annos is not None:
        results = pd.concat([annos, results], axis=1)
    return results


def slide_caption(
    wsi: WSIData,
    prompt: list[str],
    feature_key,
    *,
    agg_key: str = None,
    agg_by: str | Sequence[str] = None,
    max_length: int = 100,
    model: str = "prism",
    device: str = None,
):
    """
    Generate captions for the slide.

    Parameters
    ----------
    wsi : :class:`wsidata.WSIData`
        The WSI data object.
    prompt : list of str
        The text instruction to generate the caption.
    feature_key : str
        The slide features to be used.
    agg_key : str
        The aggregation key
    agg_by : str or list of str
        The aggregation keys that were used to create the slide features.
    max_length : int
        The maximum length of the generated caption.
    model : {"prism"}
        The caption generation model to use.
    device : str
        The device to use for inference. If None, the default device will be used.

    """

    if device is None:
        device = get_torch_device()

    from lazyslide.models.multimodal import Prism

    model = Prism()
    model.to(device)

    agg_info, annos = _get_agg_info(
        wsi,
        feature_key,
        agg_key=agg_key,
        agg_by=agg_by,
    )

    captions = []

    for ix, lat in enumerate(agg_info["latents"]):
        lat = torch.tensor(lat).unsqueeze(0).to(device)
        caption = model.caption(
            lat,
            prompt=prompt,
            max_length=max_length,
        )
        captions.append(caption)

    results = pd.DataFrame(
        {
            "caption": captions,
        }
    )
    if annos is not None:
        results = pd.concat([annos, results], axis=1)
    return results
