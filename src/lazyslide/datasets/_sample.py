import warnings
from pathlib import Path

from huggingface_hub import hf_hub_download
from wsidata import open_wsi

from lazyslide._utils import find_stack_level

REGISTRY = [
    "sample.svs",
    "GTEX-1117F-0526.svs",
    "lung_carcinoma.ndpi",
]


def _load_dataset(slide_file, zarr_file, with_data=True, pbar=False):
    if pbar:
        warnings.warn(
            "pbar is deprecated in datasets",
            DeprecationWarning,
            stacklevel=find_stack_level(),
        )

    slide = hf_hub_download(
        "RendeiroLab/LazySlide-data", slide_file, repo_type="dataset"
    )
    slide_zarr = None
    if with_data:
        slide_zarr_zip = hf_hub_download(
            "RendeiroLab/LazySlide-data", zarr_file, repo_type="dataset"
        )
        slide_zarr = Path(slide_zarr_zip.replace(".zip", ""))
        # Unzip the zarr file if it is a zip file
        # But only if it is not already unzipped
        if not slide_zarr.exists():
            from zipfile import ZipFile

            with ZipFile(slide_zarr_zip, "r") as zip_ref:
                zip_ref.extractall(slide_zarr.parent)
    return open_wsi(slide, store=str(slide_zarr) if with_data else None, pbar=pbar)


def sample(with_data: bool = True, pbar: bool = False):
    """
    Load a small sample slide (~1.9 MB).

    Source: https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/CMU-1-Small-Region.svs

    Parameters
    ----------
    with_data : bool, default: True
        Whether to load the associated zarr storage data.
    pbar : bool, default: False
        Whether to show the progress bar.

    """
    return _load_dataset(
        "sample.svs", "sample.zarr.zip", with_data=with_data, pbar=pbar
    )


def gtex_artery(with_data: bool = True, pbar: bool = False):
    """
    A GTEX artery slide.

    Source: https://gtexportal.org/home/histologyPage, GTEX-1117F-0526

    Parameters
    ----------
    with_data : bool, default: True
        Whether to load the associated zarr storage data.
    pbar : bool, default: False
        Whether to show the progress bar.

    """
    return _load_dataset(
        "GTEX-1117F-0526.svs",
        "GTEX-1117F-0526.zarr.zip",
        with_data=with_data,
        pbar=pbar,
    )


def gtex_small_intestine(with_data: bool = True, pbar: bool = False):
    """
    A small GTEX artery slide for testing purposes.

    Source: https://gtexportal.org/home/histologyPage, GTEX-11DXX-1626.svs

    Parameters
    ----------
    with_data : bool, default: True
        Whether to load the associated zarr storage data.
    pbar : bool, default: False
        Whether to show the progress bar.

    """
    return _load_dataset(
        "GTEX-11DXX-1626.svs",
        "GTEX-11DXX-1626.zarr.zip",
        with_data=with_data,
        pbar=pbar,
    )


def lung_carcinoma(with_data: bool = True, pbar: bool = False):
    """
    A lung carcinoma slide.

    Source: https://idr.openmicroscopy.org/webclient/img_detail/9846318/?dataset=10801

    Parameters
    ----------
    with_data : bool, default: True
        Whether to load the associated zarr storage data.
    pbar : bool, default: False
        Whether to show the progress bar.

    """

    return _load_dataset(
        "lung_carcinoma.ndpi", "lung_carcinoma.zarr.zip", with_data=with_data, pbar=pbar
    )
