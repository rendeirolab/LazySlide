from .openslide import OpenSlideReader


def get_reader(reader: str):
    readers = {"openslide": None}

    try:
        import openslide

        readers["openslide"] = OpenSlideReader
    except (ModuleNotFoundError, OSError) as _:
        pass

    reader_candidates = ["openslide"]
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
