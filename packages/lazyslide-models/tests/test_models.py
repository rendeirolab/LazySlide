"""
Unified model test suite.

One parametrised test function per model capability.  Adding a new model to
the registry (via ``@register``) automatically includes it in the appropriate
tests — no changes to this file required.

Device is configured via ``--device`` CLI flag (default: cpu).
Gated models are skipped unless HF credentials are present; they also carry
the ``gated`` mark so they can be filtered with ``-m 'not gated'``.

Usage examples
--------------
pytest tests/test_models.py                              # CPU, all non-gated
pytest tests/test_models.py -m 'not gated'               # explicit filter
pytest tests/test_models.py --device=cuda                # GPU
pytest tests/test_models.py --device=mps                 # Apple Silicon
pytest tests/test_models.py -k uni                       # single model
pytest tests/test_models.py --skip-models=histoplus,sam  # manual exclusions
pytest tests/test_models.py -k segmentation              # one task
"""

from __future__ import annotations

import pytest
import torch
from conftest import models_for_task
from contracts import VALIDATOR
from inputs import INPUT_FACTORY
from lazyslide_models import MODEL_REGISTRY
from lazyslide_models.base import ModelTask

# ── Shared image-prep helper ──────────────────────────────────────────────────


def _prepare_image(model, image):
    """Apply model transform if present; otherwise return the raw image.

    Models with ``get_transform() is None`` handle preprocessing internally
    (e.g. via their own HuggingFace processor).  Passing a pre-converted
    tensor to those models would bypass their processor and cause errors.
    """
    transform = model.get_transform()
    if transform is None:
        # Return raw numpy image — the model's encode_image handles preprocessing
        return image
    t = transform(image)
    # Some transforms already return a batched tensor; add batch dim if not
    if isinstance(t, torch.Tensor) and t.ndim == 3:
        t = t.unsqueeze(0)
    return t


# ═══════════════════════════════════════════════════════════════════════════════
# encode_image  —  vision + multimodal
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "model_name",
    models_for_task("vision") + models_for_task("multimodal"),
)
def test_encode_image(model_name: str, load_model) -> None:
    """encode_image() must return a 2-D float Tensor (B, D)."""
    model = load_model(model_name)
    inp = INPUT_FACTORY[ModelTask.vision](model)
    img = _prepare_image(model, inp.image)
    out = model.encode_image(img)
    VALIDATOR[ModelTask.vision](out)


# ═══════════════════════════════════════════════════════════════════════════════
# encode_image_dense  —  ViT models that expose patch-level features
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("model_name", models_for_task("vision"))
def test_encode_image_dense(model_name: str, load_model) -> None:
    """encode_image_dense() must return a 3-D float Tensor (B, N_patches, D)."""
    model = load_model(model_name)
    if not hasattr(model, "encode_image_dense"):
        pytest.skip("model does not implement encode_image_dense")
    if model.get_transform() is None:
        pytest.skip(
            "model uses internal processor; encode_image_dense not testable via raw image"
        )
    inp = INPUT_FACTORY[ModelTask.vision](model)
    img = _prepare_image(model, inp.image)
    out = model.encode_image_dense(img)
    assert isinstance(out, torch.Tensor), "encode_image_dense must return Tensor"
    assert out.ndim == 3, f"expected (B, N, D) tensor, got shape {tuple(out.shape)}"
    assert out.is_floating_point(), f"expected float dtype, got {out.dtype}"


# ═══════════════════════════════════════════════════════════════════════════════
# encode_text  —  multimodal
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("model_name", models_for_task("multimodal"))
def test_encode_text(model_name: str, load_model) -> None:
    """encode_image + encode_text must return matching (B, D) float Tensors."""
    model = load_model(model_name)
    inp = INPUT_FACTORY[ModelTask.multimodal](model)
    img = _prepare_image(model, inp.image)
    img_emb = model.encode_image(img)
    txt_emb = model.encode_text(inp.texts)
    VALIDATOR[ModelTask.multimodal](img_emb, txt_emb)


# ═══════════════════════════════════════════════════════════════════════════════
# segment  —  segmentation
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("model_name", models_for_task("segmentation"))
def test_segment(model_name: str, load_model) -> None:
    """segment() must return a dict with keys from the canonical set."""
    model = load_model(model_name)
    inp = INPUT_FACTORY[ModelTask.segmentation](model)
    transform = model.get_transform()
    if transform is not None:
        img = transform(inp.image)
        if img.ndim == 3:
            img = img.unsqueeze(0)
    else:
        img = torch.from_numpy(inp.image).unsqueeze(
            0
        )  # keep uint8 for models that want it
    out = model.segment(img)
    VALIDATOR[ModelTask.segmentation](out)


# ═══════════════════════════════════════════════════════════════════════════════
# encode_slide  —  slide_encoder
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("model_name", models_for_task("slide_encoder"))
def test_encode_slide(model_name: str, load_model) -> None:
    """encode_slide() must return a float Tensor, 1-D or 2-D."""
    model = load_model(model_name)
    inp = INPUT_FACTORY[ModelTask.slide_encoder](model)
    # Add batch dim: (1, N, D) embeddings and (1, N, 2) coords
    out = model.encode_slide(
        inp.embeddings.unsqueeze(0),
        coords=inp.coords.unsqueeze(0),
    )
    VALIDATOR[ModelTask.slide_encoder](out)


# ═══════════════════════════════════════════════════════════════════════════════
# predict  —  tile_prediction + cv_feature
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "model_name",
    models_for_task("tile_prediction") + models_for_task("cv_feature"),
)
def test_predict(model_name: str, load_model) -> None:
    """predict() must return a dict of numpy arrays."""
    model = load_model(model_name)
    # task may be a list; normalise to the primary task for input lookup
    raw_task = MODEL_REGISTRY[model_name].task
    task = raw_task[0] if isinstance(raw_task, list) else raw_task
    inp = INPUT_FACTORY[task](model)
    transform = model.get_transform()
    if transform is not None:
        img = transform(inp.image)
        if isinstance(img, torch.Tensor) and img.ndim == 3:
            img = img.unsqueeze(0)
    else:
        img = inp.image  # cv_feature models accept raw numpy
    out = model.predict(img)
    VALIDATOR[task](out)


# ═══════════════════════════════════════════════════════════════════════════════
# style transfer
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("model_name", models_for_task("style_transfer"))
def test_style_transfer(model_name: str, load_model) -> None:
    """predict() must return a float Tensor, 3-D or 4-D."""
    model = load_model(model_name)
    inp = INPUT_FACTORY[ModelTask.style_transfer](model)
    img = _prepare_image(model, inp.image)
    with torch.inference_mode():
        out = model.predict(img)
    VALIDATOR[ModelTask.style_transfer](out)


# ═══════════════════════════════════════════════════════════════════════════════
# image generation
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("model_name", models_for_task("image_generation"))
def test_image_generation(model_name: str, load_model) -> None:
    """generate() must return non-None."""
    model = load_model(model_name)
    with torch.inference_mode():
        out = model.generate()
    VALIDATOR[ModelTask.image_generation](out)
