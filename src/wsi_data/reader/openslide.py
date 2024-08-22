from pathlib import Path
from typing import Union

from .base import ReaderBase, convert_image


class OpenSlideReader(ReaderBase):
    """
    Use OpenSlide to interface with image files.

    Depends on `openslide-python <https://openslide.org/api/python/>`_
    which wraps the `openslide <https://openslide.org/>`_ C library.

    Parameters
    ----------
    file : str or Path
        Path to image file on disk

    """

    name = "openslide"

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
        img = self.reader.read_region((x, y), level, (int(width), int(height)))
        return convert_image(img)

    def get_level(self, level):
        level = self.translate_level(level)
        img = self.reader.read_region(
            (0, 0), level, self.reader.level_dimensions[level]
        )
        return convert_image(img)

    def get_thumbnail(self, size, **kwargs):
        sx, sy = self.properties.shape
        if size > sx or size > sy:
            raise ValueError("Requested thumbnail size is larger than the image")
        # The size is only the maximum size
        if sx > sy:
            size = (size, int(size * sy / sx))
        else:
            size = (int(size * sx / sy), size)

        img = self.reader.get_thumbnail(size)
        return convert_image(img)

    def detach_reader(self):
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    def create_reader(self):
        from openslide import OpenSlide

        self._reader = OpenSlide(self.file)

    @property
    def reader(self):
        if self._reader is None:
            self.create_reader()
        return self._reader
