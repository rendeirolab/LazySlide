from lazy_imports import try_import

from lazyslide.loader import FeatureExtractionDataset

with try_import() as _import:
    from torch.utils.data import DataLoader


class FeatureExtractionLoader(DataLoader):
    def __init__(self, wsi, resize=None, **kwargs):
        dataset = FeatureExtractionDataset(wsi, resize=resize)
        super().__init__(dataset, **kwargs)
