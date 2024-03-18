from __future__ import annotations

import warnings
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from numbers import Integral
from pathlib import Path
from typing import Sequence, Mapping

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from .cv_mods import TissueDetectionHE
from .io import H5ZSFile
from .plotting import SlideViewer
from .utils import (
    get_reader,
    TileOps,
    get_split_image_indices,
    filter_tiles,
    create_tiles_coords_index,
    create_tiles,
)


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
            h5_file = self.image.with_suffix(".h5zs")

        self.reader_options = {} if reader_options is None else reader_options
        self.h5_file: H5ZSFile = H5ZSFile(h5_file)
        self._reader_class = get_reader(reader)
        if self._reader_class is None:
            raise RuntimeError("No reader is available")
        self._reader = None
        self._tile_ops = self.h5_file.get_tile_ops()
        self._tile_coords = self.h5_file.get_coords()
        self._masks = {}
        self._masks_level = {}
        self._contours, self._holes = None, None
        self._table = None
        self._fields = {}

    def __repr__(self):
        h, w = self.metadata.level_shape[0]
        repr_str = (
            f"WSI Image {w}px * {h}px: {self.image}\n"
            f"  Levels: {self.metadata.n_level}\n"
            f"  MPP: {self.metadata.mpp}\n"
            f"  Magnification: {int(self.metadata.magnification)}X\n"
            f"H5: {self.h5_file.file}\n"
            f"Reader: {self._reader_class.name}\n"
        )
        if self.table is not None:
            format_table_keys = ", ".join([f'"{k}"' for k in self.table.columns])
            repr_str += f"Table keys: \n  {format_table_keys}\n"
        if len(self.fields) > 0:
            format_field_keys = ", ".join([f'"{k}"' for k in self.fields.keys()])
            repr_str += f"Feature Fields: \n  {format_field_keys}"

        return repr_str

    @property
    def reader(self):
        if self._reader is None:
            self._reader = self._reader_class(self.image, **self.reader_options)
        return self._reader

    def detach_handler(self):
        if self._reader is not None:
            self._reader.detach_handler()

    @property
    def metadata(self):
        return self.reader.metadata

    @property
    def tiles_coords(self):
        return self._tile_coords

    @property
    def n_tiles(self):
        if self._tile_coords is None:
            return 0
        return len(self._tile_coords)

    @property
    def tile_ops(self):
        return self._tile_ops

    @property
    def contours(self):
        if self._contours is None:
            self._contours, self._holes = self.h5_file.get_contours_holes()
        return self._contours

    @property
    def holes(self):
        if self._holes is None:
            self._contours, self._holes = self.h5_file.get_contours_holes()
        return self._holes

    @property
    def table(self):
        if self._table is None:
            self._table = self.h5_file.get_table()
        return self._table

    @property
    def fields(self):
        if len(self._fields) == 0:
            field_keys = self.h5_file.get_available_feature_fields()
            for field in field_keys:
                self._fields[field] = self.h5_file.get_feature_field(field)
        return self._fields

    def get_tiles_coords(self):
        return self.tiles_coords

    def get_mask(self, name):
        if name in self._masks:
            return self._masks[name], self._masks_level[name]
        return self.h5_file.get_mask(name)

    @property
    def has_tiles(self):
        return self.h5_file.has_tiles()

    # def shuffle_tiles(self, seed=0):
    #     rng = np.random.default_rng(seed)
    #     rng.shuffle(self._tile_coords)

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

        self._masks[name] = mask
        self._masks_level[name] = level
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
        self._contours = contours
        self._holes = holes
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
        tile_filter=None,
        save=True,
    ):
        """
        Parameters
        ----------
        tile_px : int or (int, int)
            Size of tile, either an integer or a tuple in (Width, Height)
        stride_px : int or (int, int)
            Size of stride between each tile,
            either an integer or a tuple in (Width, Height).
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
            tile_w, tile_h = (tile_px, tile_px)
        elif isinstance(tile_px, Sequence):
            tile_w, tile_h = (tile_px[0], tile_px[1])
        else:
            raise TypeError(
                f"input tile_px {tile_px} invalid. "
                f"Either (W, H), or a single integer for square tiles."
            )

        if stride_px is None:
            stride_h, stride_w = tile_h, tile_w
        elif isinstance(stride_px, Integral):
            stride_h, stride_w = (stride_px, stride_px)
        elif isinstance(stride_px, Sequence):
            stride_h, stride_w = (stride_px[0], stride_px[1])
        else:
            raise TypeError(
                f"input stride {stride_px} invalid. "
                f"Either (W, H), or a single integer."
            )

        use_mask = True
        mask, mask_level = self.get_mask(mask_name)
        if mask is None:
            # Try to use contours instead
            if self._contours is None:
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
            ops_tile_w = int(tile_w * downsample)
            ops_tile_h = int(tile_h * downsample)
            ops_stride_w = int(stride_w * downsample)
            ops_stride_h = int(stride_h * downsample)
        else:
            ops_tile_w, ops_tile_h = tile_w, tile_h
            ops_stride_w, ops_stride_h = stride_w, stride_h

        # Generate coords
        image_shape = self.metadata.level_shape[ops_level]

        # Filter coords based on mask
        # TODO: Consider create tiles based on the
        #       bbox of different components

        if use_mask:
            tiles_coords = create_tiles(
                image_shape, ops_tile_w, ops_tile_h, ops_stride_w, ops_stride_h, pad=pad
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
                int(ops_tile_w * ratio),
                int(ops_tile_h * ratio),
                filter_bg=background_fraction,
            )
            tiles_coords = tiles_coords[use_tiles].copy()

        else:
            rect_coords, rect_indices = create_tiles_coords_index(
                image_shape, ops_tile_w, ops_tile_h, ops_stride_w, ops_stride_h, pad=pad
            )
            if len(self._contours) == 0:
                is_tiles = np.zeros(len(rect_coords), dtype=np.bool_)
            else:
                points = rect_coords
                is_in = []
                for c in self._contours:
                    polytest = [
                        cv2.pointPolygonTest(c, (float(x), float(y)), measureDist=False)
                        for x, y in points
                    ]
                    is_in.append(np.array(polytest) == 1)

                if len(self._holes) > 0:
                    for c in self._holes:
                        polytest = [
                            cv2.pointPolygonTest(
                                c, (float(x), float(y)), measureDist=False
                            )
                            for x, y in points
                        ]
                        is_in.append(np.array(polytest) == -1)

                is_tiles = np.asarray(is_in).sum(axis=0) == 1
            # The number of points for each tiles inside contours
            good_tiles = is_tiles[rect_indices].sum(axis=1) >= tile_pts
            # Select only the top_left corner
            tiles_coords = rect_coords[rect_indices[good_tiles, 0]].copy()

        # Run tile filter
        if tile_filter is not None:
            masks = []

            for c in tiles_coords:
                left, top = c
                patch = self.get_patch(left, top, tile_w, tile_h, level=0)
                masks.append(tile_filter.filter(patch))
            tiles_coords = tiles_coords[masks].copy()

        self.new_tiles(
            tiles_coords,
            width=tile_w,
            height=tile_h,
            level=ops_level,
            mpp=mpp,
            downsample=downsample,
            ops_height=ops_tile_h,
            ops_width=ops_tile_w,
            overwrite=True,
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
        format="left-top",
        overwrite=False,
        preserve_table=True,
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
        format : str, {'left-top', 'top-left'}
            The input format of your tiles coordination
        save : bool
            Whether to back up the tiles coordination on disk in h5

        Returns
        -------

        """
        if not overwrite:
            if self.has_tiles:
                raise ValueError(
                    "Tiles already exist, please use overwrite=True to overwrite it."
                    "Reset tiles will clean up tables and feature fields,"
                    "If you want to preserve the table and feature fields, "
                    "please use preserve_table=True"
                )
        tiles_coords = np.asarray(tiles_coords, dtype=np.uint32)
        if format == "top-left":
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
        self._tile_coords = tiles_coords

        if save:
            self.h5_file.set_coords(tiles_coords)
            self.h5_file.set_tile_ops(self._tile_ops)
            if not preserve_table:
                self.h5_file.delete_table()
                for field in self.fields.keys():
                    self.h5_file.delete_feature_field(field)

    def new_table(self, table: pd.DataFrame, save=True):
        assert len(table) == len(
            self._tile_coords
        ), "Table must have the same length as tiles coords"
        self._table = table.copy()
        if save:
            self.h5_file.set_table(table)

    def new_feature(self, field: str, value: np.ndarray, save=True):
        assert value.ndim < 3, "Value must be 1D or 2D"
        assert len(value) == len(
            self._tile_coords
        ), "Value must have the same length as tiles coords"
        self._fields[field] = value.copy()
        if save:
            self.h5_file.set_feature_field(field, value)

    # def report(self):
    #     if self._tile_ops is not None:
    #         print(
    #             f"Generate tiles with mpp={self._tile_ops.mpp}, WSI mpp={self.metadata.mpp}\n"
    #             f"Total tiles: {len(self.tiles_coords)}"
    #             f"Use mask: '{self._tile_ops.mask_name}'\n"
    #             f"Generated Tiles in px (H, W): ({self._tile_ops.height}, {self._tile_ops.width})\n"
    #             f"WSI Tiles in px (H, W): ({self._tile_ops.ops_height}, {self._tile_ops.ops_width}) \n"
    #             f"Down sample ratio: {self._tile_ops.downsample}"
    #         )

    def plot_tissue(
        self,
        level=-1,
        max_size=1000,
        tiles=False,
        contours=False,
        contour_color="green",
        hole_color="blue",
        linewidth=1,
        show_origin=True,
        title=None,
        ax=None,
        save=None,
        savefig_kws=None,
    ):
        level = self.reader.translate_level(level)
        image_arr = self.reader.get_level(level)
        scale_factor = 1 / self.metadata.level_downsample[level]

        if title is None:
            title = self.image.name
        viewer = SlideViewer(
            image_arr, max_size=max_size, ax=ax, scale_factor=scale_factor
        )
        viewer.add_tissue(title=title)
        if show_origin:
            viewer.add_origin()
        if tiles:
            if isinstance(tiles, bool):
                if not self.has_tiles:
                    warnings.warn("No tile is created")
                else:
                    viewer.add_tiles(self.tiles_coords, self.tile_ops, alpha=0.5)
            else:
                if isinstance(tiles, tuple) and len(tiles) == 4:
                    x1, y1, w, h = tiles
                    tile_ops = deepcopy(self.tile_ops)
                    tile_ops.ops_height = h
                    tile_ops.ops_width = w
                    viewer.add_tiles(
                        np.array([[x1, y1]]),
                        tile_ops,
                        alpha=0.5,
                    )
                else:
                    raise ValueError(
                        "tiles must be a tuple of (x1, y1, w, h) or a boolean"
                    )

        if contours:
            if self.contours is None:
                warnings.warn("No contours is created")
            else:
                viewer.add_contours_holes(
                    self.contours,
                    self.holes,
                    contour_color=contour_color,
                    hole_color=hole_color,
                    linewidth=linewidth,
                )

        if save is not None:
            savefig_kws = {} if savefig_kws is None else savefig_kws
            viewer.fig.savefig(save, **savefig_kws)

        return viewer.ax

    def plot_table(
        self,
        columns: str | Sequence[str],
        level=-1,
        max_size=1000,
        ncols=5,
        wspace=0.3,
        hspace=0.05,
        cmap="viridis",
        norm=None,
        palette=None,
        alpha=None,
        show_origin=True,
        show_tissue=True,
        title=None,
        save=None,
        savefig_kws=None,
    ):
        if self.table is None:
            raise ValueError("No table is created")
        if isinstance(columns, str):
            columns = [columns]
        tb_columns = set(self.table.columns)
        for column in columns:
            if column not in tb_columns:
                raise ValueError(f"Column '{column}' does not exist in table")
        if not self.has_tiles:
            raise ValueError("No tile is created")

        # smart layout for subplots
        if ncols > len(columns):
            ncols = len(columns)
        nrows = len(columns) // ncols
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(ncols * 6, nrows * 6),
            gridspec_kw={"wspace": wspace, "hspace": hspace},
        )

        level = self.reader.translate_level(level)
        image_arr = self.reader.get_level(level)
        scale_factor = 1 / self.metadata.level_downsample[level]
        viewer = SlideViewer(
            image_arr, max_size=max_size, scale_factor=scale_factor, fig=fig
        )

        axes = axes.flat if isinstance(axes, np.ndarray) else [axes]
        for i, (column, ax) in enumerate(zip(columns, axes)):
            plot_arr = self._table[column].values
            if show_tissue:
                viewer.add_tissue(ax=ax)
            if show_origin:
                viewer.add_origin(ax=ax)
            if title is None:
                t = column
            elif isinstance(title, Mapping):
                t = title[column]
            else:
                t = title[i]
            viewer.add_tiles(
                self.tiles_coords,
                self.tile_ops,
                value=plot_arr,
                title=t,
                cmap=cmap,
                norm=norm,
                palette=palette,
                alpha=alpha,
                ax=ax,
            )

        if save is not None:
            savefig_kws = {} if savefig_kws is None else savefig_kws
            viewer.save(save, **savefig_kws)

    def plot_feature(
        self,
        field,
        index=0,
        level=-1,
        max_size=1000,
        ncols=5,
        wspace=0.3,
        hspace=0.05,
        cmap="viridis",
        norm=None,
        palette=None,
        alpha=None,
        show_origin=True,
        show_tissue=True,
        save=None,
        savefig_kws=None,
    ):
        if self.fields is None:
            raise ValueError("Feature Fields are empty")
        if field not in self.fields.keys():
            raise ValueError(f"Field {field} does not exist.")
        if not self.has_tiles:
            raise ValueError("No tile is created")
        if isinstance(index, Integral):
            index = [index]

        field_data = self.fields[field]

        # smart layout for subplots
        if ncols > len(index):
            ncols = len(index)
        nrows = len(index) // ncols
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(ncols * 6, nrows * 6),
            gridspec_kw={"wspace": wspace, "hspace": hspace},
        )

        level = self.reader.translate_level(level)
        image_arr = self.reader.get_level(level)
        scale_factor = 1 / self.metadata.level_downsample[level]
        viewer = SlideViewer(
            image_arr, max_size=max_size, scale_factor=scale_factor, fig=fig
        )

        axes = axes.flat if isinstance(axes, np.ndarray) else [axes]
        for i, (ix, ax) in enumerate(zip(index, axes)):
            if field_data.ndim == 1:
                plot_arr = field_data
            else:
                plot_arr = field_data[:, index].flatten()
            if show_tissue:
                viewer.add_tissue(ax=ax)
            if show_origin:
                viewer.add_origin(ax=ax)
            t = f"{field}_{ix}"
            viewer.add_tiles(
                self.tiles_coords,
                self.tile_ops,
                value=plot_arr,
                title=t,
                cmap=cmap,
                norm=norm,
                palette=palette,
                alpha=alpha,
                ax=ax,
            )

        if save is not None:
            savefig_kws = {} if savefig_kws is None else savefig_kws
            viewer.save(save, **savefig_kws)

    def plot_mask(
        self,
        name="tissue",
        max_size=1000,
        ax=None,
        save=None,
        savefig_kws=None,
    ):
        image_arr, _ = self.get_mask(name)
        if image_arr is None:
            raise NameError(f"Cannot draw non-exist mask with name '{name}'")

        viewer = SlideViewer(image_arr, max_size=max_size, ax=ax)
        viewer.add_mask(image_arr, cmap="gray")
        if save is not None:
            savefig_kws = {} if savefig_kws is None else savefig_kws
            viewer.fig.savefig(save, **savefig_kws)
        return viewer.ax

    def get_patch(self, left, top, width, height, level=0, **kwargs):
        return self.reader.get_patch(
            int(left), int(top), int(width), int(height), level=level, **kwargs
        )

    def get_tile_by_index(self, index):
        return self._tile_coords[index]
