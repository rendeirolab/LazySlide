import torch
from torch.utils.data import Dataset
from torchvision.transforms.v2 import ToImage, ToDtype, Normalize, Compose, Resize
from numbers import Number

from lazyslide.normalizer import ColorNormalizer


def compose_transform(resize=None,
                      color_normalize=None,
                      feature_extraction=False,
                      ):
    if feature_extraction:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    before = [ToImage(), ToDtype(dtype=torch.float32)]
    middle = []
    after = [Normalize(mean=mean, std=std)]
    if resize is not None:
        middle.append(Resize(size=resize))
    if color_normalize is not None:
        middle.append(ColorNormalizer(method=color_normalize))

    return Compose(before + middle + after)


class FeatureExtractionDataset(Dataset):

    def __init__(self,
                 wsi,
                 transform=None,
                 resize=None,
                 color_normalize=None,
                 ):
        self.wsi = wsi
        self.tiles_coords = self.wsi.h5_file.get_coords()
        self.tile_ops = self.wsi.h5_file.get_tile_ops()
        if transform is not None:
            self.transform = transform
        else:
            if resize is not None:
                resize_to = resize
            elif self.tile_ops.downsample != 1:
                resize_to = (int(self.tile_ops.height), int(self.tile_ops.width))
            else:
                resize_to = None
            self.transform = compose_transform(resize=resize_to,
                                               color_normalize=color_normalize,
                                               feature_extraction=True)

    def __len__(self):
        return len(self.tiles_coords)

    def __getitem__(self, idx):
        coords = self.tiles_coords[idx]
        x, y = coords
        image = self.wsi.get_patch(y, x,
                                   self.tile_ops.ops_width,
                                   self.tile_ops.ops_height,
                                   self.tile_ops.level)
        return self.transform(image)
