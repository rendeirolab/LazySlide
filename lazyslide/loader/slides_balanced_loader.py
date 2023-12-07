import warnings
from copy import deepcopy
from collections import deque

import numpy as np
from torch.utils.data import Dataset, Sampler, DataLoader
from torchvision.transforms.v2 import Resize, Compose, ToImage

from lazyslide.utils import pairwise
from .dataset import compose_transform


class Slide:

    def __init__(self,
                 n_tiles,
                 start_index,
                 end_index,
                 shuffle=True,
                 seed=0
                 ):
        self.n_tiles = n_tiles
        self.start_index = start_index
        self.end_index = end_index
        if shuffle:
            rng = np.random.default_rng(seed)
            self.pool = rng.choice(
                np.arange(start_index, end_index), n_tiles,
                # Each index can only be sample once
                replace=False)
        else:
            self.pool = np.arange(start_index, start_index + n_tiles)
        self.pool = deque(self.pool)

    def __len__(self):
        return self.n_tiles

    def get_tile(self):
        if len(self.pool) == 0:
            return None
        return self.pool.pop()


class SlidesDataset(Dataset):
    # TODO: Allow both tile labels or slide labels to be passed in

    def __init__(self,
                 wsi_list,
                 resize=None,
                 antialias=False,
                 color_normalize=None,
                 transform=None,
                 max_taken=None,
                 shuffle_slides=True,
                 shuffle_tiles=True,
                 seed=0
                 ):
        try:
            from ncls import NCLS
        except ImportError:
            raise ModuleNotFoundError("Install NCLS with `pip install ncls`.")

        self.resize_transform = None
        if transform is not None:
            self.transform = transform
        else:
            self.transform = compose_transform(
                resize, color_normalize=color_normalize, feature_extraction=False
            )

        if resize is None:
            self.resize_transform = []
            for wsi in wsi_list:
                if wsi.tile_ops.downsample != 1:
                    resize_to = (wsi.tile_ops.height, wsi.tile_ops.width)
                    self.resize_transform.append(
                        Compose([ToImage(), Resize(size=resize_to, antialias=antialias)])
                    )
                else:
                    self.resize_transform.append(None)

        rng = np.random.default_rng(seed)

        self.proxy_ix = np.arange(len(wsi_list))
        if shuffle_slides:
            rng.shuffle(self.proxy_ix)

        self.seed = seed
        self.shuffle_tiles = shuffle_tiles
        self.wsi_list = wsi_list
        self.wsi_n_tiles = [len(wsi_list[i].tiles_coords) for i in self.proxy_ix]
        self.ix_slides = np.insert(np.cumsum(self.wsi_n_tiles), 0, 0)

        self.ixs = []
        self.starts = []
        self.ends = []

        for slide_id, (start, end) in enumerate(pairwise(self.ix_slides)):
            self.ixs.append(slide_id)
            self.starts.append(start)
            self.ends.append(end)
        self.ncls = NCLS(
            np.array(self.starts, dtype=int),
            np.array(self.ends, dtype=int),
            np.array(self.ixs, dtype=int),
        )

        self.max_taken = max_taken

    def __len__(self):
        return sum(self.wsi_n_tiles)

    def __getitem__(self, ix):
        ix = int(ix)
        _, _, slide_ix = next(self.ncls.find_overlap(ix, ix + 1))
        tile_ix = ix - self.starts[slide_ix]
        wsi = self.wsi_list[self.proxy_ix[slide_ix]]

        # change here how to get the coordinate
        top, left = wsi.tiles_coords[tile_ix]
        tile_ops = wsi.tile_ops
        img = wsi.get_patch(left, top, tile_ops.ops_width,
                            tile_ops.ops_height, tile_ops.level)
        if self.resize_transform is not None:
            resize_ops = self.resize_transform[slide_ix]
            if resize_ops is not None:
                img = resize_ops(img)

        return self.transform(img)

    def get_sampler_slides(self):
        slides = []
        less_than_max_taken = []
        less_n_tiles = []

        for slide_ix, (n_tiles, start_index, end_index) in enumerate(
                zip(self.wsi_n_tiles, self.starts, self.ends)):
            if self.max_taken is not None:
                if n_tiles > self.max_taken:
                    n_tiles = self.max_taken
                if n_tiles < self.max_taken:
                    less_than_max_taken.append(self.wsi_list[slide_ix].image)
                    less_n_tiles.append(n_tiles)

            slides.append(Slide(n_tiles, start_index, end_index,
                                shuffle=self.shuffle_tiles))

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

    def __init__(self, slides, batch_size, drop_last=False):
        super().__init__()
        self.slides = slides
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __len__(self):
        return sum([len(s) for s in self.slides]) // self.batch_size

    def __iter__(self):

        _iter_slides = deepcopy(self.slides)

        exhaust_slides = []
        batch = []

        while True:

            for slide in _iter_slides:
                t = slide.get_tile()
                # If tile can be acquired
                if t is not None:
                    batch.append(t)
                # If not, the slide is exhausted
                else:
                    exhaust_slides.append(slide)

                if len(batch) == self.batch_size:
                    yield batch
                    batch = []

                for s in exhaust_slides:
                    _iter_slides.remove(s)
                    exhaust_slides = []

            if len(_iter_slides) == 0:
                # There are two situations
                # 1. All tiles equally divided to each batch
                # 2. Not equally divided
                if len(batch) > 0 and not self.drop_last:
                    yield batch
                return


class SlidesBalancedLoader(DataLoader):
    """The loader will ensure for each batch,
    the tiles are from different slides
    """

    def __init__(self, wsi_list,
                 batch_size=1,
                 resize=None,
                 antialias=False,
                 color_normalize=None,
                 transform=None,
                 max_taken=None,
                 drop_last=False,
                 shuffle_slides=True,
                 shuffle_tiles=True,
                 seed=0,
                 **kwargs,
                 ):
        dataset = SlidesDataset(wsi_list,
                                resize=resize,
                                antialias=antialias,
                                color_normalize=color_normalize,
                                transform=transform,
                                max_taken=max_taken,
                                shuffle_slides=shuffle_slides,
                                shuffle_tiles=shuffle_tiles,
                                seed=seed,
                                )
        sampler = SlidesSampler(dataset.get_sampler_slides(),
                                batch_size=batch_size,
                                drop_last=drop_last)

        super().__init__(
            dataset=dataset,
            batch_sampler=sampler,
            **kwargs,
        )
