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
