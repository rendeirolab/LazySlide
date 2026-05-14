import torch
import torch.nn as nn
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
            wsi_small, model=TIMM_MODEL, load_kws=dict(pretrained=False)
        )

    def test_timm_vit_model(self, wsi_small):
        zs.tl.feature_extraction(
            wsi_small, model=TIMM_VIT_MODEL, dense=True, load_kws=dict(pretrained=False)
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
