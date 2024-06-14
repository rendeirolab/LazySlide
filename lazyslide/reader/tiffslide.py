from .openslide import OpenSlideReader


class TiffSlideReader(OpenSlideReader):
    """
    Use OpenSlide to interface with image files.

    Depends on `openslide-python <https://openslide.org/api/python/>`_
    which wraps the `openslide <https://openslide.org/>`_ C library.

    Parameters
    ----------
    file : str or Path
        Path to image file on disk

    """

    name = "tiffslide"

    def create_reader(self):
        from tiffslide import TiffSlide

        self._reader = TiffSlide(self.file)
