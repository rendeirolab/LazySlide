from pathlib import Path
from typing import Union

from .base import ReaderBase, convert_image


# NOTE: This is a placeholder for the actual implementation
#      of the CuCIM reader. It's not tested and will not work.
# TODO: Implement the CuCIM reader on GPU-available machine
class CuCIMReader(ReaderBase):
    """
    Use CuCIM to interface with image files.

    See `CuCIM <https://github.com/rapidsai/cucim>`_ for more information.

    Parameters
    ----------
    file : str or Path
        Path to image file on disk

    """

    def __init__(
        self,
        file: Union[Path, str],
        **kwargs,
    ):
        self.file = str(file)
        self.create_reader()
        self.set_properties(self._reader.properties)

    def get_region(
        self,
        x,
        y,
        width,
        height,
        level: int = 0,
        **kwargs,
    ):
        level = self.translate_level(level)
        img = self.reader.read_region(
            (x, y),
            (int(width), int(height)),
            level=level,
        )
        return convert_image(img)

    def get_level(self, level):
        level = self.translate_level(level)
        img = self.reader.read_region(
            (0, 0), level, self.reader.level_dimensions[level]
        )
        return convert_image(img)

    def detach_reader(self):
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    def create_reader(self):
        from cucim import CuImage

        self._reader = CuImage(self.file)

    @property
    def reader(self):
        if self._reader is None:
            self.create_reader()
        return self._reader
