"""Mock models for fast, offline testing.

These replace heavy models (instanseg, nulite, plip, rosie, prism, grandqc)
so tests validate pipeline logic without downloading weights.
"""

from __future__ import annotations

from typing import Any, Dict, Self, Tuple

import numpy as np
import torch
import torch.nn as nn
from lazyslide_models.base import (
    ImageTextModel,
    ModelBase,
    ModelTask,
    SegmentationModel,
    StyleTransferModel,
)


# ---------------------------------------------------------------------------
# Cell segmentation mock (replaces instanseg)
# ---------------------------------------------------------------------------
class MockCellSegmentationModel(SegmentationModel):
    """Returns instance_map with 3 synthetic cells per tile."""

    def __init__(self, **kwargs):
        self.model = nn.Identity()

    def get_transform(self):
        from torchvision.transforms.v2 import Compose, ToDtype, ToImage

        return Compose([ToImage(), ToDtype(dtype=torch.float32, scale=False)])

    def segment(self, image) -> Dict[str, Any]:
        B, C, H, W = image.shape
        instance_maps = torch.zeros(B, H, W, dtype=torch.long)
        # Place 3 small cells in center region (away from edges for filtering)
        centers = [(H // 4, W // 4), (H // 2, W // 2), (3 * H // 4, 3 * W // 4)]
        for idx, (cy, cx) in enumerate(centers, start=1):
            r = min(H, W) // 20  # small radius
            y_lo, y_hi = max(0, cy - r), min(H, cy + r)
            x_lo, x_hi = max(0, cx - r), min(W, cx + r)
            instance_maps[:, y_lo:y_hi, x_lo:x_hi] = idx
        return {"instance_map": instance_maps}

    def supported_outputs(self) -> Tuple[str, ...]:
        return ("instance_map",)

    @classmethod
    def check_input_tile(cls, tile_spec) -> bool:
        return True


# ---------------------------------------------------------------------------
# Cell type segmentation mock (replaces nulite)
# ---------------------------------------------------------------------------
class MockCellTypeSegmentationModel(SegmentationModel):
    """Returns instance_map + class_map with 6-class NuLite-compatible output."""

    _CLASS_MAPPING = {
        0: "Background",
        1: "Neoplastic",
        2: "Inflammatory",
        3: "Connective",
        4: "Dead",
        5: "Epithelial",
    }

    def __init__(self, **kwargs):
        self.model = nn.Identity()

    def get_transform(self):
        from torchvision.transforms.v2 import Compose, ToDtype, ToImage

        return Compose([ToImage(), ToDtype(dtype=torch.float32, scale=True)])

    def segment(self, image) -> Dict[str, Any]:
        B, C, H, W = image.shape
        n_classes = 6
        instance_maps = np.zeros((B, H, W), dtype=np.int64)
        class_maps = np.zeros((B, n_classes, H, W), dtype=np.float32)

        # Place 3 cells, each with different class
        centers = [(H // 4, W // 4), (H // 2, W // 2), (3 * H // 4, 3 * W // 4)]
        for b in range(B):
            for idx, (cy, cx) in enumerate(centers, start=1):
                r = min(H, W) // 20
                y_lo, y_hi = max(0, cy - r), min(H, cy + r)
                x_lo, x_hi = max(0, cx - r), min(W, cx + r)
                instance_maps[b, y_lo:y_hi, x_lo:x_hi] = idx
                # Assign each cell a different class (1-indexed, skip background)
                class_id = (idx % (n_classes - 1)) + 1
                class_maps[b, class_id, y_lo:y_hi, x_lo:x_hi] = 0.9
                class_maps[b, 0, y_lo:y_hi, x_lo:x_hi] = 0.1  # low background prob

        return {
            "instance_map": instance_maps,
            "class_map": class_maps,
        }

    def supported_outputs(self) -> Tuple[str, ...]:
        return "instance_map", "class_map"

    @staticmethod
    def get_classes():
        return {
            0: "Background",
            1: "Neoplastic",
            2: "Inflammatory",
            3: "Connective",
            4: "Dead",
            5: "Epithelial",
        }

    @classmethod
    def check_input_tile(cls, tile_spec) -> bool:
        return True


# ---------------------------------------------------------------------------
# Semantic segmentation mock (replaces grandqc-artifact)
# ---------------------------------------------------------------------------
class MockSemanticSegmentationModel(SegmentationModel):
    """Returns probability_map (B, 8, H, W) for artifact segmentation."""

    def __init__(self, **kwargs):
        self.model = nn.Identity()

    def get_transform(self):
        from torchvision.transforms.v2 import Compose, Normalize, ToDtype, ToImage

        return Compose(
            [
                ToImage(),
                ToDtype(dtype=torch.float32, scale=True),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def segment(self, image) -> Dict[str, Any]:
        B, C, H, W = image.shape
        n_classes = 8
        prob_map = torch.zeros(B, n_classes, H, W)
        # Class 1 (Normal Tissue) gets high probability everywhere
        prob_map[:, 1, :, :] = 0.8
        # Class 2 (Fold) gets a small region with high prob
        prob_map[:, 2, H // 4 : H // 2, W // 4 : W // 2] = 0.9
        return {"probability_map": prob_map}

    def supported_outputs(self) -> Tuple[str, ...]:
        return ("probability_map",)

    @classmethod
    def check_input_tile(cls, tile_spec) -> bool:
        return True


# ---------------------------------------------------------------------------
# Image-text model mock (replaces plip)
# ---------------------------------------------------------------------------
class MockImageTextModel(ImageTextModel):
    """Mock PLIP-like model. encode_image/encode_text return deterministic tensors."""

    _EMBED_DIM = 512

    def __init__(self, **kwargs):
        self.model = nn.Linear(3, self._EMBED_DIM)  # dummy parameter holder
        self.model.eval()

    def get_transform(self):
        # Real PLIP returns None (uses processor inside encode_image)
        from torchvision.transforms.v2 import Compose, Resize, ToDtype, ToImage

        return Compose(
            [
                ToImage(),
                ToDtype(dtype=torch.float32, scale=True),
                Resize(size=(224, 224), antialias=False),
            ]
        )

    @torch.inference_mode()
    def encode_image(self, image, *args, **kwargs):
        if isinstance(image, torch.Tensor):
            B = image.shape[0]
        elif isinstance(image, (list, tuple)):
            B = len(image)
        else:
            B = 1
        gen = torch.Generator().manual_seed(42)
        emb = torch.randn(B, self._EMBED_DIM, generator=gen)
        return emb

    @torch.inference_mode()
    def encode_text(self, text, *args, **kwargs):
        if isinstance(text, str):
            text = [text]
        _ = len(text)
        # Deterministic but different per-text
        embeddings = []
        for i, t in enumerate(text):
            gen = torch.Generator().manual_seed(hash(t) % (2**31))
            embeddings.append(torch.randn(1, self._EMBED_DIM, generator=gen))
        result = torch.cat(embeddings, dim=0)
        result = torch.nn.functional.normalize(result, p=2, dim=-1)
        return result


# ---------------------------------------------------------------------------
# Style transfer mock (replaces rosie for virtual staining)
# ---------------------------------------------------------------------------
_ROSIE_MARKERS = [
    "DAPI",
    "CD45",
    "CD68",
    "CD14",
    "PD1",
    "FoxP3",
    "CD8",
    "HLA-DR",
    "PanCK",
    "CD3e",
    "CD4",
    "aSMA",
    "CD31",
    "Vimentin",
    "CD45RO",
    "Ki67",
    "CD20",
    "CD11c",
    "Podoplanin",
    "PDL1",
    "GranzymeB",
    "CD38",
    "CD141",
    "CD21",
    "CD163",
    "BCL2",
    "LAG3",
    "EpCAM",
    "CD44",
    "ICOS",
    "GATA3",
    "Gal3",
    "CD39",
    "CD34",
    "TIGIT",
    "ECad",
    "CD40",
    "VISTA",
    "HLA-A",
    "MPO",
    "PCNA",
    "ATM",
    "TP63",
    "IFNg",
    "Keratin8/18",
    "IDO1",
    "CD79a",
    "HLA-E",
    "CollagenIV",
    "CD66",
]


class MockStyleTransferModel(StyleTransferModel):
    """Mock ROSIE-like model returning (B, 50) predictions."""

    _name = "rosie"

    def __init__(self, **kwargs):
        self.model = nn.Identity()

    @property
    def name(self) -> str:
        return self._name

    def get_transform(self):
        from torchvision.transforms.v2 import (
            Compose,
            Normalize,
            Resize,
            ToDtype,
            ToImage,
        )

        return Compose(
            [
                ToImage(),
                ToDtype(dtype=torch.float32, scale=True),
                Resize(size=(224, 224), antialias=False),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    @torch.inference_mode()
    def predict(self, image):
        B = image.shape[0]
        # Non-zero values so post-processing doesn't produce all-zeros
        return torch.rand(B, 50) * 10 + 1

    def get_channel_names(self) -> Tuple[str, ...]:
        return _ROSIE_MARKERS

    def check_input_tile(self, mpp, size_x=None, size_y=None) -> bool:
        return True


# ---------------------------------------------------------------------------
# Prism mock (replaces prism for zero-shot + slide encoding)
# ---------------------------------------------------------------------------
class MockPrismModel(ModelBase):
    """Mock Prism model for zero-shot scoring and slide encoding."""

    task = [ModelTask.multimodal, ModelTask.slide_encoder]

    def __init__(self, **kwargs):
        self._device = "cpu"
        self.model = nn.Identity()

    def to(self, device) -> Self:
        self._device = device if isinstance(device, str) else str(device)
        return self

    @property
    def device(self):
        return self._device

    @torch.inference_mode()
    def encode_slide(self, embeddings, coords=None, **kwargs) -> dict:
        """Returns dict with image_embedding and image_latents."""
        B = embeddings.shape[0]
        embed_dim = 512
        n_latents = 16
        return {
            "image_embedding": torch.randn(B, embed_dim),
            "image_latents": torch.randn(B, n_latents, embed_dim),
        }

    @torch.inference_mode()
    def score(self, slide_embedding, prompts: list[list[str]]):
        """Returns softmax probabilities over prompt classes."""
        n_classes = len(prompts)
        B = slide_embedding.shape[0]
        # Deterministic logits
        logits = torch.arange(n_classes, dtype=torch.float32).unsqueeze(0).expand(B, -1)
        return torch.softmax(logits, dim=-1)
