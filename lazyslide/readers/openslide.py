import openslide
from .base import ReaderBase


class OpenSlideBackend(ReaderBase):
    """
    Use OpenSlide to interface with image files.

    Depends on `openslide-python <https://openslide.org/api/python/>`_ which wraps the `openslide <https://openslide.org/>`_ C library.

    Args:
        filename (str): path to image file on disk
    """

    def __init__(self, filename):
        self.filename = filename
        self.slide = openslide.open_slide(filename=filename)
        self.level_count = self.slide.level_count

    def __repr__(self):
        return f"OpenSlideBackend('{self.filename}')"

    def extract_region(self, location, size, level=None):
        """
        Extract a region of the image

        Args:
            location (Tuple[int, int]): Location of top-left corner of tile (i, j)
            size (Union[int, Tuple[int, int]]): Size of each tile. May be a tuple of (height, width) or a
                single integer, in which case square tiles of that size are generated.
            level (int): level from which to extract chunks. Level 0 is highest resolution.

        Returns:
            np.ndarray: image at the specified region
        """
        # verify args
        if isinstance(size, int):
            size = (size, size)
        else:
            assert (
                isinstance(size, tuple)
                and all([isinstance(a, int) for a in size])
                and len(size) == 2
            ), f"Input size {size} not valid. Must be an integer or a tuple of two integers."
        if level is None:
            level = 0
        else:
            assert isinstance(level, int), f"level {level} must be an integer"
            assert (
                level < self.slide.level_count
            ), f"input level {level} invalid for a slide with {self.slide.level_count} levels"

        # openslide read_region expects (x, y) coords, so need to switch order for compatibility with pathml (i, j)
        i, j = location

        # openslide read_region() uses coords in the level 0 reference frame
        # if we are reading tiles from a higher level, need to convert to level 0 frame by multiplying by scale factor
        # see: https://github.com/Dana-Farber-AIOS/pathml/issues/240
        coord_scale_factor = int(self.slide.level_downsamples[level])
        i *= coord_scale_factor
        j *= coord_scale_factor

        h, w = size
        region = self.slide.read_region(location=(j, i), level=level, size=(w, h))
        region_rgb = pil_to_rgb(region)
        return region_rgb

    def get_image_shape(self, level=0):
        """
        Get the shape of the image at specified level.

        Args:
            level (int): Which level to get shape from. Level 0 is highest resolution. Defaults to 0.

        Returns:
            Tuple[int, int]: Shape of image at target level, in (i, j) coordinates.
        """
        assert isinstance(level, int), f"level {level} invalid. Must be an int."
        assert (
            level < self.slide.level_count
        ), f"input level {level} invalid for slide with {self.slide.level_count} levels total"
        j, i = self.slide.level_dimensions[level]
        return i, j

    def get_thumbnail(self, size):
        """
        Get a thumbnail of the slide.

        Args:
            size (Tuple[int, int]): the maximum size of the thumbnail

        Returns:
            np.ndarray: RGB thumbnail image
        """
        thumbnail = self.slide.get_thumbnail(size)
        thumbnail = pil_to_rgb(thumbnail)
        return thumbnail

    def generate_tiles(self, shape=3000, stride=None, pad=False, level=0):
        """
        Generator over tiles.

        Padding works as follows:
        If ``pad is False``, then the first tile will start flush with the edge of the image, and the tile locations
        will increment according to specified stride, stopping with the last tile that is fully contained in the image.
        If ``pad is True``, then the first tile will start flush with the edge of the image, and the tile locations
        will increment according to specified stride, stopping with the last tile which starts in the image. Regions
        outside the image will be padded with 0.
        For example, for a 5x5 image with a tile size of 3 and a stride of 2, tile generation with ``pad=False`` will
        create 4 tiles total, compared to 6 tiles if ``pad=True``.

        Args:
            shape (int or tuple(int)): Size of each tile. May be a tuple of (height, width) or a single integer,
                in which case square tiles of that size are generated.
            stride (int): stride between chunks. If ``None``, uses ``stride = size`` for non-overlapping chunks.
                Defaults to ``None``.
            pad (bool): How to handle tiles on the edges. If ``True``, these edge tiles will be zero-padded
                and yielded with the other chunks. If ``False``, incomplete edge chunks will be ignored.
                Defaults to ``False``.
            level (int, optional): For slides with multiple levels, which level to extract tiles from.
                Defaults to 0 (highest resolution).

        Yields:
            pathml.core.tile.Tile: Extracted Tile object
        """
        assert isinstance(shape, int) or (
            isinstance(shape, tuple) and len(shape) == 2
        ), f"input shape {shape} invalid. Must be a tuple of (H, W), or a single integer for square tiles"
        if isinstance(shape, int):
            shape = (shape, shape)
        assert (
            stride is None
            or isinstance(stride, int)
            or (isinstance(stride, tuple) and len(stride) == 2)
        ), f"input stride {stride} invalid. Must be a tuple of (stride_H, stride_W), or a single int"
        if level is None:
            level = 0
        assert isinstance(level, int), f"level {level} invalid. Must be an int."
        assert (
            level < self.slide.level_count
        ), f"input level {level} invalid for slide with {self.slide.level_count} levels total"

        if stride is None:
            stride = shape
        elif isinstance(stride, int):
            stride = (stride, stride)

        i, j = self.get_image_shape(level=level)

        stride_i, stride_j = stride

        # calculate number of expected tiles
        # check for tile shape evenly dividing slide shape to fix https://github.com/Dana-Farber-AIOS/pathml/issues/305
        if pad and i % stride_i != 0:
            n_tiles_i = i // stride_i + 1
        else:
            n_tiles_i = (i - shape[0]) // stride_i + 1
        if pad and j % stride_j != 0:
            n_tiles_j = j // stride_j + 1
        else:
            n_tiles_j = (j - shape[1]) // stride_j + 1

        for ix_i in range(n_tiles_i):
            for ix_j in range(n_tiles_j):
                coords = (int(ix_i * stride_i), int(ix_j * stride_j))
                # get image for tile
                tile_im = self.extract_region(location=coords, size=shape, level=level)
                yield pathml.core.tile.Tile(image=tile_im, coords=coords)


def pil_to_rgb(image_array_pil):
    """
    Convert PIL RGBA Image to numpy RGB array
    """
    image_array_rgba = np.asarray(image_array_pil)
    image_array = cv2.cvtColor(image_array_rgba, cv2.COLOR_RGBA2RGB).astype(np.uint8)
    return image_array
