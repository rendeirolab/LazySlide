from .openslide import OpenSlideReader
from .cucim import CuCIMReader
from .tiffslide import TiffSlideReader


def get_reader(reader: str):
    readers = {"openslide": None, "cucim": None, "tiffslide": None}

    try:
        import openslide

        readers["openslide"] = OpenSlideReader
    except (ModuleNotFoundError, OSError) as _:
        pass

    try:
        import tiffslide

        readers["tiffslide"] = TiffSlideReader
    except (ModuleNotFoundError, OSError) as _:
        pass

    try:
        import cucim

        readers["cucim"] = CuCIMReader
    except (ModuleNotFoundError, OSError) as _:
        pass

    reader_candidates = ["openslide", "tiffslide", "cucim"]
    if reader == "auto":
        for i in reader_candidates:
            reader = readers.get(i)
            if reader is not None:
                return reader
    elif reader not in reader_candidates:
        raise ValueError(
            f"Requested reader not available, " f"must be one of {reader_candidates}"
        )
    else:
        return readers[reader]
