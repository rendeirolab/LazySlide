from __future__ import annotations

from numbers import Integral
from pathlib import Path
from typing import Iterable

import numpy as np

from lazyslide.cv_mods import TissueDetectionHE
from lazyslide.readers.base import ReaderBase
from lazyslide.utils import get_reader


class WSI:

    def __int__(self,
                image: Path | str,
                h5_file: Path | str = None
                ):
        self.image = Path(image)

        if h5_file is None:
            h5_file = self.image.with_suffix(".coords.h5")

        self.h5_file = h5_file
        reader = get_reader()
        self.reader: ReaderBase = reader(self.image)
        self.n_level

    def __repr__(self):
        return (f"WSI(image={self.image}, "
                f"h5_file={self.h5_file})")

    def apply(self, transform, level=-1) -> np.ndarray:
        pass

    def segment_tissue(self, level=-1, **kwargs) -> np.ndarray:
        if level == -1:
            level = self.reader.metadata.n_level

        self.reader.get_level(level)
        TissueDetectionHE(**kwargs)

    def extract_tiles(self, tile_px, tile_um, tolerance_um=.05):
        """

        Parameters
        ----------
        tile_px
        tile_um
        tolerance_um : If the MPP is not very different from the
            actually one, we can tolerate this so resize operation
            for every tile is not needed.

        Returns
        -------

        """
        pass

    @staticmethod
    def img_tile_coordinates(image_shape, tile_px=256, stride=None, pad=False):
        """
        Calculate tile coordinates.

        Padding works as follows:
        If ``pad is False``, then the first tile will start flush with the edge of the image, and the tile locations
        will increment according to specified stride, stopping with the last tile that is fully contained in the image.
        If ``pad is True``, then the first tile will start flush with the edge of the image, and the tile locations
        will increment according to specified stride, stopping with the last tile which starts in the image. Regions
        outside the image will be padded with 0.
        For example, for a 5x5 image with a tile size of 3 and a stride of 2, tile generation with ``pad=False`` will
        create 4 tiles total, compared to 6 tiles if ``pad=True``.

        Args:
            image_shape (tuple(int, int)) : The shape of the image in (height, width), like numpy array
            tile_px (int or tuple(int)): Size of each tile. Maybe a tuple of (height, width) or a single integer,
                in which case square tiles of that size are generated.
            stride (int): stride between chunks. If ``None``, uses ``stride = size`` for non-overlapping chunks.
                Defaults to ``None``.
            pad (bool): How to handle tiles on the edges. If ``True``, these edge tiles will be zero-padded
                and yielded with the other chunks. If ``False``, incomplete edge chunks will be ignored.
                Defaults to ``False``.

        Return:
            Tile coordinates in a list

        """

        if isinstance(tile_px, Integral):
            tile_px = (tile_px, tile_px)
        elif isinstance(tile_px, Iterable):
            tile_px = (tile_px[0], tile_px[1])
        else:
            raise TypeError(f"input tile_px {tile_px} invalid. "
                            f"Either (H, W), or a single integer for square tiles.")

        if stride is None:
            stride = tile_px
        elif isinstance(stride, Integral):
            stride = (stride, stride)
        elif isinstance(stride, Iterable):
            stride = (stride[0], stride[1])
        else:
            raise TypeError(f"input stride {tile_px} invalid. "
                            f"Either (H, W), or a single integer.")

        height, width = image_shape

        stride_height, stride_width = stride

        # calculate number of expected tiles
        if pad and height % stride_height != 0:
            n_tiles_height = height // stride_height + 1
        else:
            n_tiles_height = (height - tile_px[0]) // stride_height + 1
        if pad and width % stride_width != 0:
            n_tiles_width = width // stride_width + 1
        else:
            n_tiles_width = (width - tile_px[1]) // stride_width + 1

        coordinates = list()
        for ix_height in range(n_tiles_height):
            for ix_width in range(n_tiles_width):
                coords = (int(ix_height * stride_height), int(ix_width * stride_width))
                coordinates.append(coords)

        return coordinates
