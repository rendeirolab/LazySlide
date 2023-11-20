from dataclasses import dataclass, field
from numbers import Number


class BackendBase:

    def get_patch(self):
        """Get a patch from image with top-left corner"""

        raise NotImplemented

    def get_cell(self):
        """Get a patch from image with center"""

        raise NotImplemented

    def get_metadata(self):
        raise NotImplemented

    @staticmethod
    def img_tile_coordinates(image_shape, tile_px=3000, stride=None, pad=False):
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
            tile_px (int or tuple(int)): Size of each tile. May be a tuple of (height, width) or a single integer,
                in which case square tiles of that size are generated.
            stride (int): stride between chunks. If ``None``, uses ``stride = size`` for non-overlapping chunks.
                Defaults to ``None``.
            pad (bool): How to handle tiles on the edges. If ``True``, these edge tiles will be zero-padded
                and yielded with the other chunks. If ``False``, incomplete edge chunks will be ignored.
                Defaults to ``False``.

        Return:
            Tile coordinates in a list
        """
        if isinstance(tile_px, int) or (
            isinstance(tile_px, tuple) and len(tile_px) == 2
        ):
            raise TypeError(f"input tile_px {tile_px} invalid. Must be a tuple of (H, W), or a single integer for square tiles")
        if isinstance(tile_px, int):
            tile_px = (tile_px, tile_px)

        if (
            stride is None
            or isinstance(stride, int)
            or (isinstance(stride, tuple) and len(stride) == 2)
        ):
            raise TypeError(f"input stride {stride} invalid. Must be a tuple of (stride_H, stride_W), or a single int")
        
        if stride is None:
            stride = tile_px
        elif isinstance(stride, int):
            stride = (stride, stride)

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


@dataclass
class WSIMetaData:
    file_id: str
    mpp: field(None)
