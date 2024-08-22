from .openslide import OpenSlideReader
from .cucim import CuCIMReader
from .tiffslide import TiffSlideReader


def get_reader(reader: str = None):
    readers = {"openslide": None, "cucim": None, "tiffslide": None}
    error_stack = {"openslide": None, "cucim": None, "tiffslide": None}
    catch_error = (ModuleNotFoundError, OSError, ImportError)

    try:
        import openslide

        readers["openslide"] = OpenSlideReader
    except catch_error as e:
        error_stack["openslide"] = e

    try:
        import tiffslide

        readers["tiffslide"] = TiffSlideReader
    except catch_error as e:
        error_stack["tiffslide"] = e

    try:
        import cucim

        readers["cucim"] = CuCIMReader
    except catch_error as e:
        error_stack["cucim"] = e

    reader_candidates = ["openslide", "tiffslide", "cucim"]
    if reader is None:
        for i in reader_candidates:
            reader = readers.get(i)
            if reader is not None:
                return reader
        raise ValueError(
            f"None of the readers are available:"
            f"\nopenslide: {error_stack['openslide']}"
            f"\ntiffslide: {error_stack['tiffslide']}"
            f"\ncucim: {error_stack['cucim']}"
        )
    elif reader not in reader_candidates:
        raise ValueError(
            f"Requested reader not available, " f"must be one of {reader_candidates}"
        )
    else:
        used_reader = readers.get(reader)
        if used_reader is None:
            raise ValueError(
                f"Requested reader not available: {reader}, "
                f"following error occurred: {error_stack[reader]}"
            )
        return used_reader
