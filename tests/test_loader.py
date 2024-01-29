import torch

from conftest import import_windows_modules

import_windows_modules()

import lazyslide as zs
from lazyslide.loader import SlidesBalancedLoader, FeatureExtractionDataset


class TestLoader:
    def setup(self):
        slide = "https://github.com/camicroscope/Distro/raw/master/images/sample.svs"
        wsi = zs.WSI(slide)
        wsi.create_tissue_mask()
        wsi.create_tiles(512)
        self.wsi = wsi

    def test_balanced_loader(self):
        wsi = self.wsi
        loader = SlidesBalancedLoader(wsi_list=[wsi], shared_memory=True)

        for _ in zip(loader, range(10)):
            pass

    def test_feature_dataset(self):
        wsi = self.wsi
        dataset = FeatureExtractionDataset(wsi, resize=224, antialias=False)
        for _ in zip(dataset, range(10)):
            pass

    def test_feature_loader(self):
        wsi = self.wsi
        dataset = FeatureExtractionDataset(wsi, resize=224, antialias=False)
        loader = torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=2)
        for _ in zip(loader, range(2)):
            pass
