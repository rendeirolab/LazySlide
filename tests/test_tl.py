import lazyslide as zs


class TestFeatureExtraction:
    def test_load_model(self, wsi, torch_model_file):
        zs.pp.find_tissues(wsi)
        zs.pp.tile_tissues(wsi, 512)
        zs.tl.feature_extraction(wsi, model_path=torch_model_file)

    def test_load_jit_model(self, wsi, torch_jit_file):
        zs.tl.feature_extraction(wsi, model_path=torch_jit_file)

    def test_torch_hub_model(self, wsi):
        zs.tl.feature_extraction(wsi, model="resnet18")
