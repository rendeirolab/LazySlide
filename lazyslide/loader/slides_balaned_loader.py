import warnings
from copy import deepcopy

import numpy as np
from torch.utils.data import Dataset, Sampler, DataLoader
from torchvision.transforms.v2 import Resize, Compose, ToImage

from lazyslide.utils import pairwise
from .dataset import compose_transform


class Slide:

    def __init__(self,
                 n_tiles,
                 slide_ix,
                 start_index,
                 ):
        self.n_tiles = n_tiles
        self.ix = slide_ix
        self.start_index = start_index

    def __len__(self):
        return self.n_tiles

    def get_tile(self):
        if self.n_tiles == 0:
            return None
        self.n_tiles -= 1
        return self.start_index + self.n_tiles


class SlidesDataset(Dataset):

    def __init__(self,
                 wsi_list,
                 resize=None,
                 color_normalize=None,
                 transform=None,
                 max_taken=None,
                 ):
        try:
            from ncls import NCLS
        except ImportError:
            raise ModuleNotFoundError("Install NCLS with `pip install ncls`.")

        self.resize_transform = None
        if transform is not None:
            self.transform = transform
        else:
            self.transform = compose_transform(resize,
                                               color_normalize=color_normalize,
                                               feature_extraction=False)

        if resize is None:
            self.resize_transform = []
            for wsi in wsi_list:
                if wsi.tile_ops.downsample != 1:
                    resize_to = (wsi.tile_ops.height, wsi.tile_ops.width)
                    self.resize_transform.append(
                        Compose([ToImage(), Resize(size=resize_to)])
                    )
                else:
                    self.resize_transform.append(None)

        self.wsi_list = wsi_list
        self.wsi_n_tiles = [len(wsi.tiles_coords) for wsi in wsi_list]
        self.ix_slides = np.insert(np.cumsum(self.wsi_n_tiles), 0, 0)

        self.ixs = []
        self.starts = []
        self.ends = []
        for ix, (start, end) in enumerate(pairwise(self.ix_slides)):
            self.ixs.append(ix)
            self.starts.append(start)
            self.ends.append(end)
        self.ncls = NCLS(np.array(self.starts, dtype=int), np.array(self.ends, dtype=int),
                         np.array(self.ixs, dtype=int))

        self.max_taken = max_taken

    def __len__(self):
        return sum(self.wsi_n_tiles)

    def __getitem__(self, ix):
        ix = int(ix)
        _, _, slide_ix = next(self.ncls.find_overlap(ix, ix+1))
        tile_ix = ix - self.starts[slide_ix]
        wsi = self.wsi_list[slide_ix]

        # change here how to get the coordinate
        top, left = wsi.tiles_coords[tile_ix]
        tile_ops = wsi.tile_ops
        img = wsi.get_patch(left, top, tile_ops.ops_width, tile_ops.ops_height, tile_ops.level)
        if self.resize_transform is not None:
            resize_ops = self.resize_transform[slide_ix]
            if resize_ops is not None:
                img = resize_ops(img)

        return self.transform(img)

    def get_sampler_slides(self):
        slides = []
        less_than_max_taken = []
        less_n_tiles = []

        for slide_ix, (n_tiles, start_index) in enumerate(zip(self.wsi_n_tiles, self.starts)):
            if self.max_taken is not None:
                if n_tiles > self.max_taken:
                    n_tiles = self.max_taken
                if n_tiles < self.max_taken:
                    less_than_max_taken.append(self.wsi_list[slide_ix].image)
                    less_n_tiles.append(n_tiles)

            slides.append(Slide(n_tiles, slide_ix, start_index))

        total_less = len(less_than_max_taken)
        if total_less > 0:
            if total_less > 30:
                less_than_max_taken = less_than_max_taken[0:30]
                less_n_tiles = less_n_tiles[0:30]
            warn_stats = [f'{i}, {n} tiles' for i, n in zip(less_than_max_taken, less_n_tiles)]
            warnings.warn(f"There are {total_less} slides has less than max_taken={self.max_taken}:"
                          f"{', '.join(warn_stats)}")
        return slides


class SlidesSampler(Sampler):

    def __init__(self, slides, batch_size):
        super().__init__()
        self.slides = slides
        self.batch_size = batch_size

    def __len__(self):
        return sum([len(s) for s in self.slides])

    def __iter__(self):

        _iter_slides = deepcopy(self.slides)

        unused_slides = []
        batch = []

        while True:

            for slide in _iter_slides:
                t = slide.get_tile()
                if t is not None:
                    batch.append(t)
                else:
                    unused_slides.append(slide)

                if len(batch) == self.batch_size:
                    yield batch
                    batch = []

                for s in unused_slides:
                    _iter_slides.remove(s)
                    unused_slides = []

            if len(_iter_slides) == 0:
                yield batch
                return


class SlidesBalancedLoader(DataLoader):
    """The loader will ensure for each batch,
    the tiles are from different slides
    """

    def __init__(self, wsi_list,
                 batch_size=1,
                 resize=None,
                 color_normalize=None,
                 transform=None,
                 max_taken=None,
                 **kwargs,
                 ):
        dataset = SlidesDataset(wsi_list,
                                resize=resize,
                                color_normalize=color_normalize,
                                transform=transform,
                                max_taken=max_taken)
        sampler = SlidesSampler(dataset.get_sampler_slides(),
                                batch_size=batch_size)

        super().__init__(
            dataset=dataset,
            batch_sampler=sampler,
            **kwargs,
        )
