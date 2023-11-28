import torch
from torch.utils.data import Dataset
from torchvision.transforms.v2 import ToDtype, Normalize, Compose, Resize


class WSIDataset(Dataset):

    def __init__(self,
                 wsi,
                 transform=None,
                 run_pretrained=False):
        self.wsi = wsi
        self.tiles_coords = self.wsi.h5_file.get_coords()
        self.tile_ops = self.wsi.h5_file.get_tile_ops()

        if run_pretrained:
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
        else:
            mean = (0.5, 0.5, 0.5)
            std = (0.5, 0.5, 0.5)

        self.transform_ops = [ToDtype(torch.float32, scale=True), Normalize(mean=mean, std=std)]
        if transform is not None:
            self.transform_ops = [transform]
        if self.tile_ops.downsample != 1:
            self.transform_ops = [Resize(size=(self.tile_ops.height,
                                               self.tile_ops.width))]\
                                 + self.transform_ops
        self.transform = Compose(self.transform_ops)

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
