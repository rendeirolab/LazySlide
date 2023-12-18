from __future__ import annotations

import warnings
import weakref
from numbers import Integral
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from numba import njit

from .cv_mods import TissueDetectionHE
from .h5 import H5File
from .utils import pairwise, get_reader, TileOps


@njit
def create_tiles_top_left(
    image_shape, tile_h, tile_w, stride_h=None, stride_w=None, pad=True
):
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

    coordinates = []
    for ix_height in range(n_tiles_height):
        for ix_width in range(n_tiles_width):
            coords = (int(ix_height * stride_h), int(ix_width * stride_w))
            coordinates.append(coords)

    return np.array(coordinates, dtype=np.uint)


@njit
def filter_tiles(mask, tiles_coords, tile_h, tile_w, filter_bg=0.8):
    """Return a binary array that indicate which tile should be left

    Parameters
    ----------
    mask
    tiles_coords
    filter_bg
    tile_h,
    tile_w,
    Returns
    -------

    """
    use = []
    for x, y in tiles_coords:
        mask_region = mask[x : x + tile_h, y : y + tile_w]
        bg_ratio = np.sum(mask_region == 0) / mask_region.size
        use.append(bg_ratio < filter_bg)
    return np.array(use, dtype=np.bool_)


@njit
def create_tiles_coords_index(
    image_shape, tile_h, tile_w, stride_h=None, stride_w=None, pad=True
):
    """Create the tiles, return coordination that comprise the tiles
        and the index of points for each rect

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
    indices = list()
    for ix_height in range(n_tiles_height + 1):
        for ix_width in range(n_tiles_width + 1):
            coords = (int(ix_height * stride_h), int(ix_width * stride_w))
            coordinates.append(coords)
            if (ix_height != n_tiles_height) & (ix_width != n_tiles_width):
                ix1 = ix_height * (n_tiles_width + 1) + ix_width
                ix3 = (ix_height + 1) * (n_tiles_width + 1) + ix_width
                indices.append([ix1, ix1 + 1, ix3 + 1, ix3])

    return (np.array(coordinates, dtype=np.uint), np.array(indices, dtype=np.uint))


def get_split_image_indices(image_height, image_width, min_side=20000):
    h, w = image_height, image_width
    size = h * w
    n = min_side
    if (size > n**2) or (h > n) or (w > n):
        split_h = h > 1.5 * n
        split_w = w > 1.5 * n

        if not split_h and not split_w:
            return

        n_chunk_h = int(np.ceil(h / n))
        n_chunk_w = int(np.ceil(w / n))

        # If split, return the split chunks
        # Else, it would take the whole
        ix_h = (
            np.linspace(start=0, stop=h, num=n_chunk_h + 1, dtype=int)
            if split_h
            else [0, h]
        )
        ix_w = (
            np.linspace(start=0, stop=w, num=n_chunk_w + 1, dtype=int)
            if split_w
            else [0, w]
        )

        slices = []
        for h1, h2 in pairwise(ix_h):
            row = []
            for w1, w2 in pairwise(ix_w):
                row.append((h1, h2, w1, w2))
            slices.append(row)
        return slices


class WSI:
    """Container class for one whole-slide image

    Parameters
    ----------
    image : str or Path
        The path to the WSI file
    h5_file : str or Path
        The path to the h5 file that store the metadata
    reader : {'auto', 'openslide', 'vips', 'cucim'}
        The reader to use
    reader_options : dict
        The options to pass to the reader

    """

    def __init__(
        self,
        image: Path | str,
        h5_file: Path | str = None,
        reader="auto",  # openslide, vips, cucim
        reader_options=None,
    ):
        from .utils import check_wsi_path

        self.image = check_wsi_path(image)

        if h5_file is None:
            h5_file = self.image.with_suffix(".coords.h5")

        self.reader_options = {} if reader_options is None else reader_options
        self.h5_file: H5File = H5File(h5_file)
        self._reader_class = get_reader(reader)
        self._reader = None
        self._tile_ops = self.h5_file.get_tile_ops()
        self.contours, self.holes = self.h5_file.get_contours_holes()

    def __repr__(self):
        return (
            f"WSI(image={self.image}, "
            f"h5_file={self.h5_file.file},"
            f"reader={self._reader_class})"
        )

    @property
    def reader(self):
        if self._reader is None:
            self._reader = self._reader_class(self.image, **self.reader_options)
        return self._reader

    def detach_handler(self):
        self._reader.detach_handler()

    @property
    def metadata(self):
        return self.reader.metadata

    @property
    def tiles_coords(self):
        return self.h5_file.get_coords()

    @property
    def tile_ops(self):
        return self._tile_ops

    def get_tiles_coords(self):
        return self.tiles_coords

    def get_mask(self, name):
        return self.h5_file.get_masks(name)

    @property
    def has_tiles(self):
        return self.h5_file.has_tiles

    def shuffle_tiles(self, seed=0):
        rng = np.random.default_rng(seed)
        tiles_coords = self.tiles_coords
        rng.shuffle(tiles_coords)
        self.h5_file.set_coords(tiles_coords)
        self.h5_file.set_tile_ops(self._tile_ops)

    def move_wsi_file(self, new_path: Path) -> None:
        new_path = Path(new_path)
        if not new_path.exists():
            self.image.rename(new_path)
            self.image = new_path
        else:
            raise FileExistsError(f"File {new_path} already exists.")

    def create_mask(self, transform, name="user", level=-1, save=False):
        level = self.reader.translate_level(level)
        image = self.reader.get_level(level)
        mask = transform.apply(image)
        if save:
            self.h5_file.set_mask(name, mask, level)

    def create_tissue_mask(
        self, name="tissue", level=-1, chunk=True, chunk_at=20000, save=False, **kwargs
    ):
        """Create tissue mask using
        a preconfigured segmentation pipeline

        Parameters
        ----------
        name : str
            The name of the mask to be created
        level : int
            The slide level to work with
        chunk : bool
            Whether to split image into chunks when it's too large
        chunk_at : int
            Only chunk the image when a side of image is above this threshold
        save : bool
            Whether to save the mask into h5
        kwargs

        Returns
        -------

        """

        level = self.reader.translate_level(level)
        img_height, img_width = self.metadata.level_shape[level]
        seg = TissueDetectionHE(**kwargs)

        # If the image is too large, we will run segmentation by chunk
        split_indices = get_split_image_indices(
            img_height, img_width, min_side=chunk_at
        )
        if chunk & (split_indices is not None):
            mask = np.zeros((img_height, img_width), dtype=np.uint)
            for row in split_indices:
                for ixs in row:
                    h1, h2, w1, w2 = ixs
                    img_chunk = self.reader.get_patch(
                        w1, h1, w2 - w1, h2 - h1, level=level
                    )
                    chunk_mask = seg.apply(img_chunk)
                    mask[h1:h2, w1:w2] = chunk_mask
                    del img_chunk  # Explicitly release memory
        else:
            image = self.reader.get_level(level)
            mask = seg.apply(image)

        if save:
            self.h5_file.set_mask(name, mask, level)

    def create_tissue_contours(self, level=-1, save=False, **kwargs):
        """Contours will always return
        the version scale back to level 0"""

        level = self.reader.translate_level(level)
        kwargs = {"return_contours": True, **kwargs}
        seg = TissueDetectionHE(**kwargs)
        image = self.reader.get_level(level)
        contours, holes = seg.apply(image)
        if level != 0:
            downsample = self.metadata.level_downsample[level]
        else:
            downsample = 1
        contours = [(c * downsample).astype(int) for c in contours]
        holes = [(h * downsample).astype(int) for h in holes]
        self.contours = contours
        self.holes = holes
        if save:
            self.h5_file.set_contours_holes(contours, holes)

    def create_tiles(
        self,
        tile_px,
        stride_px=None,
        pad=False,
        mpp=None,
        tolerance=0.05,
        mask_name="tissue",
        background_fraction=0.8,
        tile_pts=3,
        errors="ignore",
        save=True,
    ):
        """
        Parameters
        ----------
        tile_px : int or (int, int)
            Size of tile, either an integer or a tuple in (Height, Width)
        stride_px : int or (int, int)
            Size of stride between each tile,
            either an integer or a tuple in (Height, Width).
            If not specific, this will be the same as tile_px,
            no overlapping between tiles.
        pad : bool
        mpp : float
            micron-per-pixel, most cases, 40X is 0.25 MPP, 20X is 0.5 MPP
            by default will extract from the level 0.
        tolerance : float
            If the downsample value is within a tolerance range,
            use the nearest available level without performing downsampling.
        mask_name : str
            which mask to use to filter tiles.
        background_fraction : float
            If a tile contain this much background, it will be filter out.
        errors : str, {'ignore', 'raise'}
            if mpp is not exist, raise error or ignore it.
        save : bool
            Whether to save the tiles coordination on disk in h5

        Returns
        -------

        """
        if isinstance(tile_px, Integral):
            tile_h, tile_w = (tile_px, tile_px)
        elif isinstance(tile_px, Iterable):
            tile_h, tile_w = (tile_px[0], tile_px[1])
        else:
            raise TypeError(
                f"input tile_px {tile_px} invalid. "
                f"Either (H, W), or a single integer for square tiles."
            )

        if stride_px is None:
            stride_h, stride_w = tile_h, tile_w
        elif isinstance(stride_px, Integral):
            stride_h, stride_w = (stride_px, stride_px)
        elif isinstance(stride_px, Iterable):
            stride_h, stride_w = (stride_px[0], stride_px[1])
        else:
            raise TypeError(
                f"input stride {stride_px} invalid. "
                f"Either (H, W), or a single integer."
            )

        use_mask = True
        mask, mask_level = self.get_mask(mask_name)
        if mask is None:
            # Try to use contours instead
            if self.contours is None:
                raise NameError(
                    f"Mask with name '{mask_name}' does not exist, "
                    f"use .create_tissue_contours() or .create_tissue_mask() "
                    f"to annotate tissue location."
                )
            else:
                use_mask = False

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
                    raise ValueError(
                        f"Cannot perform resize operation "
                        f"with reqeust mpp={mpp} on image"
                        f"mpp={self.metadata.mpp}, this will"
                        f"require up-scaling of image."
                    )
                elif downsample == 1:
                    ops_level = 0
                else:
                    for ix, level_downsample in enumerate(
                        self.metadata.level_downsample
                    ):
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

        # Generate coords
        image_shape = self.metadata.level_shape[ops_level]

        # Filter coords based on mask
        # TODO: Consider create tiles based on the
        #       bbox of different components

        if use_mask:
            tiles_coords = create_tiles_top_left(
                image_shape, ops_tile_h, ops_tile_w, ops_stride_h, ops_stride_w, pad=pad
            )
            # Map tile level to mask level
            # Only tile can be scale, mask will not be resized
            mask_downsample = self.metadata.level_downsample[mask_level]
            tile_downsample = self.metadata.level_downsample[ops_level]
            ratio = tile_downsample / mask_downsample
            down_coords = (tiles_coords * ratio).astype(np.uint32)
            use_tiles = filter_tiles(
                mask,
                down_coords,
                int(ops_tile_h * ratio),
                int(ops_tile_w * ratio),
                filter_bg=background_fraction,
            )
            tiles_coords = tiles_coords[use_tiles].copy()

        else:
            rect_coords, rect_indices = create_tiles_coords_index(
                image_shape, ops_tile_h, ops_tile_w, ops_stride_h, ops_stride_w, pad=pad
            )
            if len(self.contours) == 0:
                is_tiles = np.zeros(len(rect_coords), dtype=np.bool_)
            else:
                points = rect_coords
                is_in = []
                for c in self.contours:
                    # Coerce the point to python int and let the opencv decide the type
                    # Flip x, y beacuse it's different in opencv
                    is_in.append(
                        np.array(
                            [
                                cv2.pointPolygonTest(
                                    c, (float(y), float(x)), measureDist=False
                                )
                                for x, y in points
                            ]
                        )
                        == 1
                    )

                if len(self.holes) > 0:
                    for c in self.holes:
                        is_in.append(
                            np.array(
                                [
                                    cv2.pointPolygonTest(
                                        c, (float(y), float(x)), measureDist=False
                                    )
                                    for x, y in points
                                ]
                            )
                            == -1
                        )

                is_tiles = np.asarray(is_in).sum(axis=0) == 1
            # The number of points for each tiles inside contours
            good_tiles = is_tiles[rect_indices].sum(axis=1) >= tile_pts
            # Select only the top_left corner
            tiles_coords = rect_coords[rect_indices[good_tiles, 0]].copy()

        self.new_tiles(
            tiles_coords,
            height=tile_h,
            width=tile_w,
            level=ops_level,
            mpp=mpp,
            downsample=downsample,
            ops_height=ops_tile_h,
            ops_width=ops_tile_w,
            save=save,
        )

    def new_tiles(
        self,
        tiles_coords,
        height=None,
        width=None,
        level=0,
        mpp=None,
        downsample=None,
        ops_height=None,
        ops_width=None,
        format="top-left",
        save=True,
        **kwargs,
    ):
        """Supply new tiles to WSI

        The default coordination for tiles in top-left, you can change this
        with the `format` parameters.

        The coordination will be stored as uint32.

        Parameters
        ----------
        tiles_coords : array-like
            The coordination of tiles, in (x, y) format
        height
        width
        level
        format : str, {'top-left', 'left-top'}
            The input format of your tiles coordination
        save : bool
            Whether to back up the tiles coordination on disk in h5

        Returns
        -------

        """
        tiles_coords = np.asarray(tiles_coords, dtype=np.uint32)
        if format == "left-top":
            tiles_coords = tiles_coords[:, [1, 0]]
        height = int(height)
        width = int(width)
        if height is None and ops_height is None:
            raise ValueError("Either height or ops_height must be specified")
        if width is None and ops_width is None:
            raise ValueError("Either width or ops_width must be specified")
        if height is None:
            height = ops_height
        if width is None:
            width = ops_width
        if ops_height is None:
            ops_height = height
        if ops_width is None:
            ops_width = width

        self._tile_ops = TileOps(
            level=level,
            mpp=self.metadata.mpp if mpp is None else mpp,
            downsample=1 if downsample is None else downsample,
            height=height,
            width=width,
            ops_height=ops_height,
            ops_width=ops_width,
        )
        if save:
            self.h5_file.set_coords(tiles_coords)
            self.h5_file.set_tile_ops(self._tile_ops)

    def report(self):
        if self._tile_ops is not None:
            print(
                f"Generate tiles with mpp={self._tile_ops.mpp}, WSI mpp={self.metadata.mpp}\n"
                f"Total tiles: {len(self.tiles_coords)}"
                f"Use mask: '{self._tile_ops.mask_name}'\n"
                f"Generated Tiles in px (H, W): ({self._tile_ops.height}, {self._tile_ops.width})\n"
                f"WSI Tiles in px (H, W): ({self._tile_ops.ops_height}, {self._tile_ops.ops_width}) \n"
                f"Down sample ratio: {self._tile_ops.downsample}"
            )

    @staticmethod
    def _get_thumbnail(image_arr, size=1000):
        x_size, y_size = image_arr.shape[0:2]
        x_ratio = size / x_size
        # If the original image is smaller than requested size
        if x_ratio >= 1:
            return 1, image_arr
        y_shape = int(y_size * x_ratio)

        thumbnail = cv2.resize(image_arr, dsize=(y_shape, size))

        return x_ratio, thumbnail

    def plot_tissue(
        self,
        size=1000,
        tiles=False,
        edgecolor=".5",
        linewidth=1,
        contours=False,
        contours_color="green",
        holes_color="black",
        ax=None,
        savefig=None,
        savefig_kws=None,
    ):
        level = self._tile_ops.level if tiles else self.metadata.n_level - 1
        image_arr = self.reader.get_level(level)
        down_ratio, thumbnail = self._get_thumbnail(image_arr, size)

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
        ax.imshow(thumbnail)
        ax.set_axis_off()

        if tiles:
            if not self.has_tiles:
                print("No tile is created")
            else:
                # In matplotlib, H -> Y, W -> X, so we flip the axis
                coords = self.tiles_coords[:, ::-1] * down_ratio

                tile_h = self._tile_ops.height * down_ratio
                tile_w = self._tile_ops.width * down_ratio

                tiles = [Rectangle(t, tile_w, tile_h) for t in coords]
                collections = PatchCollection(
                    tiles, facecolor="none", edgecolor=edgecolor, lw=linewidth
                )

                ax.add_collection(collections)

        if contours:
            ratio = (1 / self.metadata.level_downsample[level]) * down_ratio
            if len(self.contours) > 0:
                for c in self.contours:
                    ax.plot(
                        c[:, 0] * ratio, c[:, 1] * ratio, lw=linewidth, c=contours_color
                    )
            if len(self.holes) > 0:
                for h in self.holes:
                    ax.plot(
                        h[:, 0] * ratio, h[:, 1] * ratio, lw=linewidth, c=holes_color
                    )

        if savefig:
            savefig_kws = {} if savefig_kws is None else savefig_kws
            save_kws = {"dpi": 150, **savefig_kws}
            fig.savefig(savefig, **save_kws)

        return ax

    def plot_mask(
        self,
        name="tissue",
        size=1000,
        ax=None,
        savefig=None,
        savefig_kws=None,
    ):
        image_arr = self.get_mask(name)
        if image_arr is None:
            raise NameError(f"Cannot draw non-exist mask with name '{name}'")

        _, thumbnail = self._get_thumbnail(image_arr, size)
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
        ax.imshow(thumbnail, cmap="gray")
        ax.set_axis_off()
        if savefig:
            savefig_kws = {} if savefig_kws is None else savefig_kws
            save_kws = {"dpi": 150, **savefig_kws}
            fig.savefig(savefig, **save_kws)
        return ax

    # def to_dataset(self, transform=None, run_pretrained=False, **kwargs):
    #     return WSIDataset(self, transform=transform, run_pretrained=run_pretrained, **kwargs)

    def get_patch(self, left, top, width, height, level=0, **kwargs):
        return self.reader.get_patch(
            int(left), int(top), int(width), int(height), level=level, **kwargs
        )

    def get_tile_by_index(self, index):
        return self.h5_file.get_one_coord_by_index(index)
