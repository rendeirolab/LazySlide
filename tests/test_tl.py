import lazyslide as zs

TIMM_MODEL = "mobilenetv3_small_050"


class TestFeatureExtraction:
    def test_load_model(self, wsi, torch_model_file):
        zs.pp.find_tissues(wsi)
        zs.pp.tile_tissues(wsi, 512)
        zs.tl.feature_extraction(wsi, model_path=torch_model_file)

    def test_load_jit_model(self, wsi, torch_jit_file):
        zs.tl.feature_extraction(wsi, model_path=torch_jit_file)

    def test_timm_model(self, wsi):
        zs.tl.feature_extraction(wsi, model=TIMM_MODEL)


class TestFeatureAggregation:
    def test_feature_aggregation(self, wsi):
        # Prepare data
        zs.pp.find_tissues(wsi)
        zs.pp.tile_tissues(wsi, 512)
        zs.tl.feature_extraction(wsi, model=TIMM_MODEL)

        # Test feature aggregation
        zs.tl.feature_aggregation(wsi, feature_key=TIMM_MODEL)
