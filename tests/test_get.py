import pytest
import lazyslide as zs


class TestGet:
    @pytest.fixture(autouse=True)
    def init_wsi(self, test_slide):
        self.wsi = zs.WSI(test_slide)
        zs.pp.find_tissue(self.wsi)
        zs.pp.tiles(self.wsi, 512)
        zs.tl.feature_extraction(self.wsi, "resnet18", pbar=False)

    @pytest.mark.parametrize("as_array", [True, False])
    def test_get_tissue_contour(self, as_array):
        for _ in zs.get.tissue_contours(self.wsi, as_array=as_array):
            pass

    @pytest.mark.parametrize("color_norm", ["macenko", "reinhard", "reinhard_modified"])
    def test_get_tissue_image_params_color_norm(self, color_norm):
        for _ in zs.get.tissue_images(self.wsi, color_norm=color_norm):
            pass

    @pytest.mark.parametrize("mask_bg", [True, False, 0, 1, 100])
    def test_get_tissue_image_params_mask_bg(self, mask_bg):
        for _ in zs.get.tissue_images(self.wsi, mask_bg=mask_bg):
            pass

    @pytest.mark.parametrize("raw", [True, False])
    def test_get_tile_images(self, raw):
        for _ in zs.get.tile_images(self.wsi, raw=raw):
            pass

    def test_get_pyramids(self):
        for _ in zs.get.pyramids(self.wsi):
            pass

    def test_get_n_tissue(self):
        assert zs.get.n_tissue(self.wsi) == 1

    def test_get_n_tiles(self):
        assert zs.get.n_tiles(self.wsi) == 5

    def test_get_features_anndata(self, tmp_path):
        adata = zs.get.features_anndata(self.wsi)
        adata_f = zs.get.features_anndata(self.wsi, feature_key="resnet18")
        adata.write_h5ad(tmp_path / "test.h5ad")
        adata_f.write_h5ad(tmp_path / "test_f.h5ad")
