"""Mock models for fast, offline testing.

These replace heavy models (instanseg, nulite, plip, rosie, prism, grandqc)
so tests validate pipeline logic without downloading weights.
"""

from __future__ import annotations

from typing import ClassVar, Self

import numpy as np
import torch
from lazyslide_models.base import (
    ImageTextModel,
    MarkerMapModel,
    ModelBase,
    ModelTask,
    SegmentationModel,
    SegmentationOutput,
    TilePredictionModel,
    VirtualStainModel,
)
from lazyslide_models.style_transfer import ROSIE
from torch import nn


# ---------------------------------------------------------------------------
# Cell segmentation mock (replaces instanseg)
# ---------------------------------------------------------------------------
class MockCellSegmentationModel(SegmentationModel):
    """Returns instance_map with 3 synthetic cells per tile. Cell 2 is a
    corner-touching "bowtie": two squares sharing a single diagonal corner.
    That is one 8-connected instance (one contour), but the contour self-touches
    at the corner, so ``buffer(0)`` later splits it into a MultiPolygon. This
    exercises the path where a single cell becomes a MultiPolygon and must stay
    a SINGLE row (1:1 with its feature), not be exploded into two."""

    _EMBED_DIM = 32

    def __init__(self, emit_tokens: bool = False, **kwargs):
        self.model = nn.Identity()
        self.emit_tokens = emit_tokens

    def get_transform(self):
        from torchvision.transforms.v2 import Compose, ToDtype, ToImage

        return Compose([ToImage(), ToDtype(dtype=torch.float32, scale=False)])

    def segment(self, image) -> SegmentationOutput:
        B, _C, H, W = image.shape
        instance_maps = torch.zeros(B, H, W, dtype=torch.long)
        r = min(H, W) // 20  # small radius, away from edges for filtering
        # Cells 1 and 3: simple square blobs (single Polygon each).
        for idx, (cy, cx) in [(1, (H // 4, W // 4)), (3, (3 * H // 4, 3 * W // 4))]:
            instance_maps[:, cy - r : cy + r, cx - r : cx + r] = idx
        # Cell 2: two squares touching only at a single diagonal corner. They are
        # 8-connected -> one instance / one contour, but the ring self-touches, so
        # buffer(0) yields a MultiPolygon.
        cy, cx = H // 2, W // 2
        instance_maps[:, cy - r : cy, cx - r : cx] = 2  # upper-left square
        instance_maps[:, cy : cy + r, cx : cx + r] = 2  # lower-right square
        token_map = None
        if self.emit_tokens:
            PH, PW = H // 16, W // 16
            gen = np.random.RandomState(0)
            token_map = gen.randn(B, self._EMBED_DIM, PH, PW).astype(np.float32)
        return SegmentationOutput(instance_map=instance_maps, patch_token_map=token_map)


# ---------------------------------------------------------------------------
# Cell type segmentation mock (replaces nulite)
# ---------------------------------------------------------------------------
class MockCellTypeSegmentationModel(SegmentationModel):
    """Returns instance_map + class_map with 6-class NuLite-compatible output."""

    classes = (
        "Background",
        "Neoplastic",
        "Inflammatory",
        "Connective",
        "Dead",
        "Epithelial",
    )

    def __init__(self, **kwargs):
        self.model = nn.Identity()

    def get_transform(self):
        from torchvision.transforms.v2 import Compose, ToDtype, ToImage

        return Compose([ToImage(), ToDtype(dtype=torch.float32, scale=True)])

    _EMBED_DIM = 64

    def segment(self, image) -> SegmentationOutput:
        B, _C, H, W = image.shape
        n_classes = 6
        instance_maps = np.zeros((B, H, W), dtype=np.int64)
        class_maps = np.zeros((B, n_classes, H, W), dtype=np.float32)
        r = min(H, W) // 20

        def paint(b, ys, xs, inst):
            instance_maps[b, ys, xs] = inst
            # Assign each cell a different class (1-indexed, skip background)
            class_id = (inst % (n_classes - 1)) + 1
            class_maps[b, class_id, ys, xs] = 0.9
            class_maps[b, 0, ys, xs] = 0.1  # low background prob

        for b in range(B):
            # Cells 1 and 3: simple square blobs.
            for inst, (cy, cx) in [
                (1, (H // 4, W // 4)),
                (3, (3 * H // 4, 3 * W // 4)),
            ]:
                paint(b, slice(cy - r, cy + r), slice(cx - r, cx + r), inst)
            # Cell 2: corner-touching bowtie -> one instance whose geometry
            # becomes a MultiPolygon after buffer(0) (see MockCellSegmentationModel).
            cy, cx = H // 2, W // 2
            paint(b, slice(cy - r, cy), slice(cx - r, cx), 2)
            paint(b, slice(cy, cy + r), slice(cx, cx + r), 2)

        # Simulate ViT patch token map [B, D, PH, PW]
        PH, PW = H // 16, W // 16  # typical ViT patch size = 16
        gen = np.random.RandomState(42)
        patch_token_map = gen.randn(B, self._EMBED_DIM, PH, PW).astype(np.float32)

        return SegmentationOutput(
            instance_map=instance_maps,
            probability_map=class_maps,
            patch_token_map=patch_token_map,
            classes=self.classes,
        )


# ---------------------------------------------------------------------------
# Semantic segmentation mock (replaces grandqc-artifact)
# ---------------------------------------------------------------------------
class MockSemanticSegmentationModel(SegmentationModel):
    """Returns probability_map (B, 8, H, W) for artifact segmentation."""

    def __init__(self, normal_prob=0.8, **kwargs):
        self.model = nn.Identity()
        self.normal_prob = normal_prob

    def get_transform(self):
        from torchvision.transforms.v2 import Compose, Normalize, ToDtype, ToImage

        return Compose(
            [
                ToImage(),
                ToDtype(dtype=torch.float32, scale=True),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def segment(self, image) -> SegmentationOutput:
        B, _C, H, W = image.shape
        n_classes = 8
        prob_map = torch.zeros(B, n_classes, H, W)
        # Class 1 (Normal Tissue) gets the same probability everywhere
        prob_map[:, 1, :, :] = self.normal_prob
        # Class 2 (Fold) gets a small region with high prob
        prob_map[:, 2, H // 4 : H // 2, W // 4 : W // 2] = 0.9
        return SegmentationOutput(probability_map=prob_map)


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
# Virtual staining mocks (replace rosie, gigatime and future stain models)
# ---------------------------------------------------------------------------


def _to_float_tensor():
    from torchvision.transforms.v2 import Compose, ToDtype, ToImage

    return Compose([ToImage(), ToDtype(dtype=torch.float32, scale=True)])


class MockRosieModel(ROSIE):
    """ROSIE by type, without its weights.

    It subclasses the real ``ROSIE`` class so ``virtual_stain`` recognises it and
    allows ROSIE's post-processing. ``__init__`` is overridden, so nothing is
    downloaded.
    """

    def __init__(self, **kwargs):
        self.model = nn.Identity()

    @property
    def name(self) -> str:
        return "rosie"

    def get_transform(self):
        return _to_float_tensor()

    @torch.inference_mode()
    def predict(self, image):
        b = image.shape[0]
        # Positive, varied values so ROSIE's contrast stretch has something to do.
        values = np.random.default_rng(0).random((b, len(self.columns))) * 10 + 1
        return dict(zip(self.columns, values.T, strict=True))


class MockTileModel(TilePredictionModel):
    """A per-tile model that is not ROSIE, with one constant per column.

    Constants make the check exact: without ROSIE's contrast stretch and median
    blur, every tile's pixel must hold exactly these values.
    """

    columns = ("focus", "tumour_prob")
    values = (0.25, 0.75)

    def __init__(self, **kwargs):
        self.model = nn.Identity()

    @property
    def name(self) -> str:
        return "fake_tile"

    def get_transform(self):
        return _to_float_tensor()

    @torch.inference_mode()
    def predict(self, image):
        b = image.shape[0]
        return {
            c: np.full(b, v, dtype=np.float32)
            for c, v in zip(self.columns, self.values)
        }


class MockMarkerMapModel(MarkerMapModel):
    """Dense marker map with one constant value per channel.

    Every tile predicts the same constants, so a correctly stitched and blended
    image equals them wherever tiles landed. ``0.25`` is chosen because a runner
    that wrongly applied its own sigmoid would turn it into ``0.562``.
    """

    channel_names = ("CD3", "CD8")
    output_range = (0.0, 1.0)
    values = (0.25, 0.75)

    def __init__(self, name: str = "fake_marker_map", output_mpp=None, **kwargs):
        self.model = nn.Identity()
        self._name = name
        self.output_mpp = output_mpp

    @property
    def name(self) -> str:
        return self._name

    def get_transform(self):
        return _to_float_tensor()

    @torch.inference_mode()
    def predict(self, image):
        b, _, h, w = image.shape
        if self.output_mpp is not None:
            # Predict onto a grid twice as coarse as the input tile.
            h, w = h // 2, w // 2
        v = torch.tensor(self.values, dtype=torch.float32).view(1, -1, 1, 1)
        return v.expand(b, -1, h, w).clone()


class MockVirtualStainModel(VirtualStainModel):
    """Two RGB stains in one pass, each a constant colour.

    ``"HER2 IHC"`` contains a space on purpose: spatialdata rejects that as an
    element name, so the runner has to turn it into a valid key.
    """

    stains = ("PAS", "HER2 IHC")
    output_range = (0.0, 1.0)
    colours = ((0.1, 0.2, 0.3), (0.6, 0.5, 0.4))

    def __init__(self, **kwargs):
        self.model = nn.Identity()

    @property
    def name(self) -> str:
        return "fake_stain"

    def get_transform(self):
        return _to_float_tensor()

    @torch.inference_mode()
    def predict(self, image):
        b, _, h, w = image.shape
        rgb = [c for colour in self.colours for c in colour]  # RGB-major
        v = torch.tensor(rgb, dtype=torch.float32).view(1, -1, 1, 1)
        return v.expand(b, -1, h, w).clone()


# ---------------------------------------------------------------------------
# Prism mock (replaces prism for zero-shot + slide encoding)
# ---------------------------------------------------------------------------
class MockPrismModel(ModelBase):
    """Mock Prism model for zero-shot scoring and slide encoding."""

    task: ClassVar[list[ModelTask]] = [
        ModelTask.multimodal,
        ModelTask.slide_encoder,
    ]

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
        """Returns dict with embeddings and latents."""
        B = embeddings.shape[0]
        embed_dim = 512
        n_latents = 16
        return {
            "embeddings": torch.randn(B, embed_dim),
            "latents": torch.randn(B, n_latents, embed_dim),
        }

    @torch.inference_mode()
    def score(self, slide_embedding, prompts: list[list[str]]):
        """Returns softmax probabilities over prompt classes."""
        n_classes = len(prompts)
        B = slide_embedding.shape[0]
        # Deterministic logits
        logits = torch.arange(n_classes, dtype=torch.float32).unsqueeze(0).expand(B, -1)
        return torch.softmax(logits, dim=-1)


class MockFeaturePredictionModel:
    """Deterministic feature predictor that records its input batches."""

    name = "mock_feature_prediction"
    features_model_name = "mock_input"
    needs_coords = False
    whole_slide = False

    def __init__(self, needs_coords=False, whole_slide=False):
        self.model = nn.Identity()
        self.batches = []
        self.coords = []
        self.device = None
        self.needs_coords = needs_coords
        self.whole_slide = whole_slide

    def to(self, device):
        self.device = device
        return self

    def get_transform(self):
        return None

    def try_compile(self, **compile_kws):
        return None

    def predict(self, features, coords=None):
        self.batches.append(features)
        if self.needs_coords:
            assert coords is not None, "needs_coords model was called without coords"
            self.coords.append(coords)
        values = np.asarray(features)
        return {
            "feature_sum": values.sum(axis=1),
            "feature_mean": values.mean(axis=1),
        }
