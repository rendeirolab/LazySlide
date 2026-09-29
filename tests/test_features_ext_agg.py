from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torchvision.transforms.v2 import Compose, Resize, ToDtype, ToImage

import lazyslide as zs

TIMM_MODEL = "test_resnet"
TIMM_VIT_MODEL = "test_vit"


class TestFeatureExtraction:
    def test_load_model(self, wsi_small, torch_model_file):
        zs.tl.feature_extraction(wsi_small, model_path=torch_model_file)
        # Test feature aggregation
        zs.tl.feature_aggregation(wsi_small, feature_key="MockNet")

    def test_load_jit_model(self, wsi_small, torch_jit_file):
        zs.tl.feature_extraction(wsi_small, model_path=torch_jit_file)

    def test_timm_model(self, wsi_small):
        zs.tl.feature_extraction(
            wsi_small, model=TIMM_MODEL, load_kws={"pretrained": False}
        )

    def test_timm_vit_model(self, wsi_small):
        zs.tl.feature_extraction(
            wsi_small, model=TIMM_VIT_MODEL, dense=True, load_kws={"pretrained": False}
        )


class _PoolModel(nn.Module):
    """Minimal model: global avg pool -> 3-dim feature."""

    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return self.pool(x).flatten(1)


class TestFeatureExtractionWithoutTileSpec:
    """Feature extraction on tiles added via add_shapes (no TileSpec)."""

    def test_basic(self, wsi_no_spec):
        transform = Compose(
            [
                ToImage(),
                ToDtype(dtype=torch.float32, scale=True),
                Resize((224, 224), antialias=False),
            ]
        )
        model = _PoolModel()
        zs.tl.feature_extraction(
            wsi_no_spec,
            model=model,
            tile_key="no_spec_tiles",
            key_added="pool_no_spec_tiles",
            transform=transform,
        )
        feat = wsi_no_spec.tables["pool_no_spec_tiles"]
        assert feat.X.shape[0] == 5
        assert feat.X.shape[1] == 3


@pytest.mark.parametrize(
    "encoder, kwarg",
    [
        ("titan", "base_tile_size"),
        ("conch_v1.5", "base_tile_size"),  # another key for the Titan class
        ("moozy", "patch_sizes"),
    ],
)
def test_slide_encoder_gets_the_level0_tile_stride(monkeypatch, encoder, kwarg):
    """TITAN grids tiles by floor((coords - min) / patch_size_lv0) and MOOZY
    measures its ALiBi distances in that spacing, so overlapping tiles must
    give them the level-0 stride. The width put neighbours on one grid cell,
    and MOOZY got nothing at all."""
    from lazyslide_models import MODEL_REGISTRY

    from lazyslide.tools._features import _encode_slide

    seen = {}

    class Spy(MODEL_REGISTRY[encoder]):
        def __init__(self):
            pass

        def to(self, device):
            return self

        def encode_slide(self, embeddings, coords=None, **kwargs):
            seen.update(kwargs)
            return {"embeddings": torch.zeros(1, 8)}

    monkeypatch.setitem(MODEL_REGISTRY, encoder, Spy)
    coords = pd.DataFrame({"minx": [0, 224, 448], "miny": [0, 0, 0]})
    half_overlap = SimpleNamespace(base_width=448, base_stride_width=224)

    _encode_slide(
        np.zeros((3, 8)), encoder, coords, device="cpu", tile_spec=half_overlap
    )

    assert seen == {kwarg: 224}
