"""
Output contract validators for each model task.

Pure functions — no model names, no if/else.
Each function raises ``AssertionError`` with a descriptive message when the
contract is violated.

The ``VALIDATOR`` dispatch table maps ``ModelTask`` → validator function.
Extend only this table when a new task is added.

Note: ``check_multimodal`` takes two arguments (img_emb, txt_emb).
All other validators take a single ``output`` argument.
"""

from __future__ import annotations

import numpy as np
import torch
from lazyslide_models.base import ModelTask

# ── Shared helpers ────────────────────────────────────────────────────────────


def _tensor(out, tag: str) -> torch.Tensor:
    assert isinstance(out, torch.Tensor), (
        f"{tag}: expected torch.Tensor, got {type(out).__name__}"
    )
    return out


def _2d_float(t: torch.Tensor, tag: str) -> None:
    assert t.ndim == 2, f"{tag}: expected 2-D tensor, got shape {tuple(t.shape)}"
    assert t.is_floating_point(), f"{tag}: expected float dtype, got {t.dtype}"


def _dict(out, tag: str) -> dict:
    assert isinstance(out, dict), f"{tag}: expected dict, got {type(out).__name__}"
    return out


# ── Per-task validators ───────────────────────────────────────────────────────


def check_vision(output) -> None:
    """encode_image → Tensor (B, D), float."""
    _2d_float(_tensor(output, "encode_image"), "encode_image")


def check_multimodal(img_emb, txt_emb) -> None:
    """encode_image + encode_text → both Tensor (B, D) float with same D."""
    _2d_float(_tensor(img_emb, "image_embedding"), "image_embedding")
    _2d_float(_tensor(txt_emb, "text_embedding"), "text_embedding")
    assert img_emb.shape[1] == txt_emb.shape[1], (
        f"Embedding dim mismatch: image={img_emb.shape[1]}, text={txt_emb.shape[1]}"
    )


def check_segmentation(output) -> None:
    """segment → dict whose keys are a subset of the four canonical output keys."""
    VALID_KEYS = {"probability_map", "instance_map", "class_map", "token_map"}
    d = _dict(output, "segment()")
    unexpected = set(d.keys()) - VALID_KEYS
    assert not unexpected, f"segment() returned unexpected keys: {unexpected}"
    for key, val in d.items():
        assert isinstance(val, (torch.Tensor, np.ndarray)), (
            f"segment()['{key}']: expected Tensor or ndarray, got {type(val).__name__}"
        )


def check_slide_encoder(output) -> None:
    """encode_slide → float Tensor, 1-D or 2-D."""
    t = _tensor(output, "encode_slide")
    assert t.is_floating_point(), f"encode_slide: expected float dtype, got {t.dtype}"
    assert t.ndim in (1, 2), (
        f"encode_slide: expected 1-D or 2-D tensor, got shape {tuple(t.shape)}"
    )


def check_tile_prediction(output) -> None:
    """predict → dict of numpy arrays."""
    d = _dict(output, "predict()")
    for key, val in d.items():
        assert isinstance(val, np.ndarray), (
            f"predict()['{key}']: expected numpy.ndarray, got {type(val).__name__}"
        )


def check_style_transfer(output) -> None:
    """predict → float Tensor, 3-D (C,H,W) or 4-D (B,C,H,W)."""
    t = _tensor(output, "StyleTransferModel.predict")
    assert t.is_floating_point(), f"style predict: expected float dtype, got {t.dtype}"
    assert t.ndim in (3, 4), (
        f"style predict: expected 3-D or 4-D tensor, got shape {tuple(t.shape)}"
    )


def check_image_generation(output) -> None:
    """generate() → non-None."""
    assert output is not None, "ImageGenerationModel.generate() returned None"


# ── Dispatch table ────────────────────────────────────────────────────────────

VALIDATOR: dict = {
    ModelTask.vision: check_vision,
    ModelTask.multimodal: check_multimodal,  # (img_emb, txt_emb)
    ModelTask.segmentation: check_segmentation,
    ModelTask.slide_encoder: check_slide_encoder,
    ModelTask.tile_prediction: check_tile_prediction,
    ModelTask.cv_feature: check_tile_prediction,
    ModelTask.feature_prediction: check_tile_prediction,
    ModelTask.style_transfer: check_style_transfer,
    ModelTask.image_generation: check_image_generation,
}
