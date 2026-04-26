"""
Mock input factories for each model task.

Pure functions — no model names, no if/else.
Input size variance is read from ``model.input_size`` (set by ``@register``).
Slide encoder embedding dimension is read from ``model.encode_dim``.

The ``INPUT_FACTORY`` dispatch table maps ``ModelTask`` → factory function.
Extend only this table when a new task is added.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch
from lazyslide_models.base import ModelTask

_RNG = np.random.default_rng(42)


# ── Input bundles ─────────────────────────────────────────────────────────────


class VisionInputs(NamedTuple):
    image: np.ndarray  # (H, W, 3) uint8


class MultimodalInputs(NamedTuple):
    image: np.ndarray  # (H, W, 3) uint8
    texts: list[str]


class SegmentationInputs(NamedTuple):
    image: np.ndarray  # (H, W, 3) uint8


class SlideEncoderInputs(NamedTuple):
    embeddings: torch.Tensor  # (N, D)
    coords: torch.Tensor  # (N, 2) float pixel coords


class TilePredictionInputs(NamedTuple):
    image: np.ndarray  # (H, W, 3) uint8


class StyleTransferInputs(NamedTuple):
    image: np.ndarray  # (H, W, 3) uint8


class ImageGenerationInputs(NamedTuple):
    pass  # generate() takes no arguments


# ── Shared helper ─────────────────────────────────────────────────────────────


def _random_image(model=None) -> np.ndarray:
    """Return a uint8 HWC numpy image at the model's preferred input size."""
    size = getattr(model, "input_size", None) or 224
    return _RNG.integers(0, 256, (size, size, 3), dtype=np.uint8)


# ── Factories ─────────────────────────────────────────────────────────────────


def make_vision(model=None) -> VisionInputs:
    return VisionInputs(_random_image(model))


def make_multimodal(model=None) -> MultimodalInputs:
    return MultimodalInputs(
        image=_random_image(model),
        texts=[
            "A histopathology tissue image.",
            "Tumor cells with high mitotic index.",
        ],
    )


def make_segmentation(model=None) -> SegmentationInputs:
    return SegmentationInputs(_random_image(model))


def make_slide_encoder(model=None) -> SlideEncoderInputs:
    """64-patch grid; embedding dim from model.encode_dim (default 768)."""
    D = getattr(model, "encode_dim", None) or 768
    N = 64
    embeddings = torch.randn(N, D)
    xs = torch.arange(8) * 256
    ys = torch.arange(8) * 256
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    coords = torch.stack([gx.flatten(), gy.flatten()], dim=1).float()
    return SlideEncoderInputs(embeddings, coords)


def make_tile_prediction(model=None) -> TilePredictionInputs:
    return TilePredictionInputs(_random_image(model))


def make_style_transfer(model=None) -> StyleTransferInputs:
    return StyleTransferInputs(_random_image(model))


def make_image_generation(model=None) -> ImageGenerationInputs:
    return ImageGenerationInputs()


# ── Dispatch table ────────────────────────────────────────────────────────────

INPUT_FACTORY: dict = {
    ModelTask.vision: make_vision,
    ModelTask.multimodal: make_multimodal,
    ModelTask.segmentation: make_segmentation,
    ModelTask.slide_encoder: make_slide_encoder,
    ModelTask.tile_prediction: make_tile_prediction,
    ModelTask.cv_feature: make_tile_prediction,
    ModelTask.feature_prediction: make_tile_prediction,
    ModelTask.style_transfer: make_style_transfer,
    ModelTask.image_generation: make_image_generation,
}
