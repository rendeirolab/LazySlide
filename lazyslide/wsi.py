from __future__ import annotations

import warnings
from numbers import Integral
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from numba import njit

from .h5 import H5File
from .cv_mods import TissueDetectionHE
from .readers.base import ReaderBase
from .torch_dataset import WSIDataset
from .utils import get_reader, TileOps


@njit
def create_tiles_coords(image_shape, tile_h, tile_w,
                        stride_h=None, stride_w=None, pad=True):
    """Create the tiles, return only coordination

    Padding works as follows:
    If ``pad is False``, then the first tile will start flush with the edge of the image, and the tile locations
    will increment according to specified stride, stopping with the last tile that is fully contained in the image.
    If ``pad is True``, then the first tile will start flush with the edge of the image, and the tile locations
    will increment according to specified stride, stopping with the last tile which starts in the image. Regions
    outside the image will be padded with 0.
    For example, for a 5x5 image with a tile size of 3 and a stride of 2, tile generation with ``pad=False`` will
    create 4 tiles total, compared to 6 tiles if ``pad=True``.

    Parameters
    ----------
    image_shape : (int, int), The shape of the image
    tile_h : int, The height of tile
    tile_w : int, The width of tile
    stride_h : int, The height of stride when move to next tile
    stride_w : int, The width of stride when move to next tile
    pad : bool, If ``True``, these edge tiles will be zero-padded
                and yielded with the other chunks.
                If ``False``, incomplete edge chunks will be ignored.
                Defaults to ``False``.

    Returns
    -------

    """
    height, width = image_shape
    if stride_h is None:
        stride_h = tile_h
    if stride_w is None:
        stride_w = tile_w

    # calculate number of expected tiles
    if pad and height % stride_h != 0:
        n_tiles_height = height // stride_h + 1
    else:
        n_tiles_height = (height - tile_h) // stride_h + 1
    if pad and width % stride_w != 0:
        n_tiles_width = width // stride_w + 1
    else:
        n_tiles_width = (width - tile_w) // stride_w + 1
    coordinates = list()
    for ix_height in range(n_tiles_height):
        for ix_width in range(n_tiles_width):
            coords = (int(ix_height * stride_h), int(ix_width * stride_w))
            coordinates.append(coords)

    return np.array(coordinates, dtype=np.uint16)


@njit
def filter_tiles(mask, tiles_coords,
                 tile_h, tile_w, filter_bg=.8):
    """

    Parameters
    ----------
    mask
    tiles_coords
    tile_h
    tile_w
    filter_bg

    Returns
    -------

    """
    filter_coords = []
    tile_size = tile_h * tile_w
    for x, y in tiles_coords:
        bg_ratio = np.sum(mask[x:x + tile_w, y:y + tile_h] == 0) / tile_size
        if bg_ratio < filter_bg:
            filter_coords.append((x, y))
    return np.array(filter_coords, dtype=np.uint16)


class WSI:

    def __init__(self,
                 image: Path | str,
                 h5_file: Path | str = None,
                 save_mask=False,
                 ):
        self.image = Path(image)

        if h5_file is None:
            h5_file = self.image.with_suffix(".coords.h5")

        self.h5_file = H5File(h5_file)
        reader = get_reader()
        self.reader: ReaderBase = reader(self.image)
        self.metadata = self.reader.metadata
        self.mask = self.h5_file.load_masks()
        self.save_mask = save_mask
        self._total_tiles = 0
        self.tiles_coords = self.h5_file.get_coords()
        self.tile_ops = self.h5_file.get_tile_ops()

    def __repr__(self):
        return (f"WSI(image={self.image}, "
                f"h5_file={self.h5_file})")

    def create_mask(self, transform, name="user", level=0):
        image = self.reader.get_level(level)
        self.mask[name] = transform.apply(image)
        if self.save_mask:
            self.h5_file.masks = self.mask
            self.h5_file.save()

    def create_tissue_mask(self, name="tissue", level=0, **kwargs):
        image = self.reader.get_level(level)
        self.mask[name] = TissueDetectionHE(**kwargs).apply(image)
        if self.save_mask:
            self.h5_file.masks = self.mask
            self.h5_file.save()

    def get_mask(self, name):
        return self.mask.get(name)

    def create_tiles(self, tile_px,
                     stride_px=None,
                     pad=False,
                     mpp=None,
                     tolerance=.05,
                     mask_name="tissue",
                     background_fraction=.8,
                     errors="ignore"):
        """

        Parameters
        ----------
        tile_px : int or (int, int), Size of tile, either an integer or a tuple in (Height, Width)
        stride_px : int or (int, int), Size of stride between each tile,
                        either an integer or a tuple in (Height, Width).
                        If not specific, this will be the same as tile_px,
                        no overlapping between tiles.
        pad : bool,
        mpp : float, micron-per-pixel, most cases, 40X is 0.25 MPP, 20X is 0.5 MPP
                    by default will extract from the level 0.
        tolerance : float, If the downsample value is within a tolerance range,
                            use the nearest available level without performing downsampling.
        mask_name : str, which mask to use to filter tiles.
        background_fraction : float, If a tile contain this much background, it will be
                                    filter out.
        errors : {'ignore', 'raise'}, if mpp is not exist, raise error or ignore it.

        Returns
        -------

        """
        if isinstance(tile_px, Integral):
            tile_h, tile_w = (tile_px, tile_px)
        elif isinstance(tile_px, Iterable):
            tile_h, tile_w = (tile_px[0], tile_px[1])
        else:
            raise TypeError(f"input tile_px {tile_px} invalid. "
                            f"Either (H, W), or a single integer for square tiles.")

        if stride_px is None:
            stride_h, stride_w = tile_h, tile_w
        elif isinstance(stride_px, Integral):
            stride_h, stride_w = (stride_px, stride_px)
        elif isinstance(stride_px, Iterable):
            stride_h, stride_w = (stride_px[0], stride_px[1])
        else:
            raise TypeError(f"input stride {stride_px} invalid. "
                            f"Either (H, W), or a single integer.")

        mask = self.mask.get(mask_name)
        if mask is None:
            raise NameError(f"Mask with name '{mask_name}' does not exist, "
                            f"use .create_tissue_mask() or .create_mask() to create mask.")

        ops_level = 0
        downsample = 1
        run_downsample = False
        if mpp is not None:
            if self.metadata.mpp is not None:
                downsample = mpp / self.metadata.mpp

                lower_ds = downsample - tolerance
                upper_ds = downsample + tolerance
                if lower_ds < 1 < upper_ds:
                    downsample = 1

                if downsample < 1:
                    raise ValueError(f"Cannot perform resize operation "
                                     f"with reqeust mpp={mpp} on image"
                                     f"mpp={self.metadata.mpp}, this will"
                                     f"require up-scaling of image.")
                elif downsample == 1:
                    ops_level = 0
                else:
                    for ix, level_downsample in enumerate(self.metadata.level_downsample):
                        if lower_ds < level_downsample < upper_ds:
                            downsample = level_downsample
                            ops_level = ix
                    else:
                        run_downsample = True
            else:
                msg = f"{self.image} does not contain MPP."
                if errors:
                    raise ValueError(msg)
                else:
                    warnings.warn(msg)

        if run_downsample:
            ops_tile_h = int(tile_h * downsample)
            ops_tile_w = int(tile_w * downsample)
            ops_stride_h = int(stride_h * downsample)
            ops_stride_w = int(stride_w * downsample)
        else:
            ops_tile_h, ops_tile_w = tile_h, tile_w
            ops_stride_h, ops_stride_w = stride_h, stride_w

        # Get image in numpy array
        image_arr = self.reader.get_level(ops_level)
        # Generate coords
        image_shape = image_arr.shape[0:2]

        tiles_coords = create_tiles_coords(
            image_shape, ops_tile_h, ops_tile_w,
            ops_stride_h, ops_stride_w, pad=pad)
        self._total_tiles = len(tiles_coords)
        # Filter coords based on mask
        self.tiles_coords = filter_tiles(
            mask, tiles_coords, ops_tile_h, ops_tile_w,
            filter_bg=background_fraction)
        self.tile_ops = TileOps(level=ops_level,
                                mpp=mpp,
                                downsample=downsample,
                                height=tile_h, width=tile_w,
                                ops_height=ops_tile_h,
                                ops_width=ops_tile_w,
                                mask_name=mask_name
                                )
        self.h5_file.set_coords(self.tiles_coords)
        self.h5_file.set_tile_ops(self.tile_ops)
        self.h5_file.save()

    def new_tiles(self, tiles_coords, height, width, level=0):
        """Supply a customized tiles"""
        self.tiles_coords = tiles_coords
        self.tile_ops = TileOps(level=level,
                                mpp=self.metadata.mpp,
                                downsample=1,
                                height=height,
                                width=width,
                                )
        self.h5_file.set_coords(self.tiles_coords)
        self.h5_file.set_tile_ops(self.tile_ops)
        self.h5_file.save()

    def report(self):
        if self.tile_ops is not None:
            return (f"Generate tile with mpp={self.tile_ops.mpp} "
                    f"Mask: '{self.tile_ops.mask_name}'.\n"
                    f"Tile in px (H, W): {self.tile_ops.height}, {self.tile_ops.width}.\n"
                    f"Resize: {self.tile_ops.downsample != 1}")
        else:
            return None

    @staticmethod
    def _get_thumbnail(image_arr, size=1000):
        x_size, y_size = image_arr.shape[0:2]
        x_ratio = size / x_size
        y_shape = int(y_size * x_ratio)

        thumbnail = cv2.resize(image_arr, dsize=(y_shape, size))

        return x_ratio, thumbnail

    def plot_tissue(self,
                    size=1000,
                    tiles=False,
                    edgecolor=".5",
                    linewidth=1,
                    ):

        image_arr = self.reader.get_level(0)
        scale_ratio, thumbnail = self._get_thumbnail(image_arr, size)

        _, ax = plt.subplots()
        ax.imshow(thumbnail)

        if tiles:
            if self.tiles_coords is None:
                print("No tile is created")
            else:
                # In matplotlib, H -> Y, W -> X, so we flip the axis
                coords = self.tiles_coords[:, ::-1] * scale_ratio

                tile_h = self.tile_ops.height * scale_ratio
                tile_w = self.tile_ops.width * scale_ratio

                tiles = [Rectangle(t, tile_w, tile_h) for t in coords]
                collections = PatchCollection(tiles, facecolor="none",
                                              edgecolor=edgecolor, lw=linewidth)

                ax.add_collection(collections)
        return ax

    def plot_mask(self,
                  name="tissue",
                  size=1000,
                  ):
        image_arr = self.mask.get(name)
        if image_arr is None:
            raise NameError(f"Cannot draw non-exist mask with name '{name}'")

        _, thumbnail = self._get_thumbnail(image_arr, size)
        _, ax = plt.subplots()
        ax.imshow(thumbnail)
        return ax

    def to_dataset(self, transform=None, run_pretrained=False):
        return WSIDataset(self, transform=transform, run_pretrained=run_pretrained)

    def get_patch(self, left, top, width, height, level=0, **kwargs):
        return self.reader.get_patch(left, top, width, height, level=level, **kwargs)
