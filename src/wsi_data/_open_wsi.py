from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import zarr
import numpy as np
from fsspec.core import url_to_fs
from anndata import AnnData
from spatialdata import read_zarr, SpatialData
from spatialdata.models import Image2DModel
from spatialdata.transformations import Scale
from rich.progress import track

from ._image import reader_datatree
from .data import WSIData
from .reader import get_reader


def open_wsi(
    wsi,
    backed_file=None,
    reader=None,
    download=True,
    cache_dir=None,
    name=None,
    attach_images=False,
    image_key="wsi",
    save_images=False,
    attach_thumbnail=True,
    thumbnail_key="wsi_thumbnail",
    thumbnail_size=2000,
    save_thumbnail=True,
    **kwargs,
):
    """Open a whole slide image.

    Parameters
    ----------
    wsi : str or Path
        The URL to whole slide image.
    backed_file : str, optional
        The backed file path, by default will create
        a zarr file with the same name as the slide file.
        You can either supply a file path or a directory.
        If a directory is supplied, the zarr file will be created in that directory.
        This is useful when you want to store all zarr files in a specific location.
    reader : str, optional
        Reader to use, by default "auto".

    Returns
    -------
    WSIData
        Whole slide image data.
    """

    # Check if the slide is a file or URL
    wsi = str(wsi)
    fs, wsi_path = url_to_fs(wsi)
    if not fs.exists(wsi_path):
        raise ValueError(f"Slide {wsi} not existed or not accessible.")
    if name is None:
        name = Path(wsi_path).name

    # Early attempt with reader
    ReaderCls = get_reader(reader)

    # TODO: When reader is not tiffslide and the slide is remote
    #       we need to download it first

    reader_obj = ReaderCls(wsi)
    wsi = Path(wsi)
    if backed_file is None:
        backed_file = wsi.with_suffix(".zarr")
    else:
        # We also support write all backed file to a directory
        backed_file_p = Path(backed_file)
        if backed_file_p.is_dir():
            zarr_name = Path(wsi).with_suffix(".zarr").name
            backed_file = backed_file_p / zarr_name
        else:
            backed_file = backed_file_p

    if backed_file.exists():
        sdata = read_zarr(backed_file)
    else:
        sdata = SpatialData()

    updated_elements = []

    if attach_images and image_key not in sdata:
        images_datatree = reader_datatree(reader_obj)
        sdata.images[image_key] = images_datatree
        if save_images:
            updated_elements.append(image_key)

    if attach_thumbnail and thumbnail_key not in sdata:
        thumbnail = reader_obj.get_thumbnail(thumbnail_size)
        thumbnail_shape = thumbnail.shape
        origin_shape = reader_obj.properties.shape
        scale_x, scale_y = (
            thumbnail_shape[0] / origin_shape[0],
            thumbnail_shape[1] / origin_shape[1],
        )

        if thumbnail is not None:
            sdata.images[thumbnail_key] = Image2DModel.parse(
                np.asarray(thumbnail).transpose(2, 1, 0),
                dims=("c", "y", "x"),
                transformations={"global": Scale([scale_x, scale_y], axes=("x", "y"))},
            )
            if save_thumbnail:
                updated_elements.append(thumbnail_key)

    slide_data = WSIData(reader_obj, sdata, backed_file)
    slide_data.add_write_elements(updated_elements)
    return slide_data


def agg_wsi(
    slides_table,
    feature_key,
    tile_key="tiles",
    wsi_col=None,
    backed_file_col=None,
    error="raise",
):
    if wsi_col is None and backed_file_col is None:
        raise ValueError("Either wsi_col or backed_file_col must be provided.")

    if backed_file_col is not None:
        backed_files = slides_table[backed_file_col]
    elif wsi_col is not None:
        backed_files = slides_table[wsi_col].apply(
            lambda x: Path(x).with_suffix(".zarr")
        )
    key = f"{feature_key}_{tile_key}_slide"

    jobs = []
    with ThreadPoolExecutor() as executor:
        for backed_f in backed_files:
            job = executor.submit(_agg_wsi, backed_f, key)
            jobs.append(job)

    n_slide = len(jobs)
    results = []
    for job in track(
        as_completed(jobs),
        total=n_slide,
        description=f"Aggretation of {n_slide} slides",
    ):
        results.append(job.result())

    X = np.vstack(results)

    # Convert index to string
    slides_table.index = slides_table.index.astype(str)
    return AnnData(X, obs=slides_table)


def _agg_wsi(f, key):
    s = read_zarr(f, selection=("labels",))
    return s.labels[key].values


class _WSICache:
    pass
