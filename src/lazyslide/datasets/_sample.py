import os
import warnings
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from wsidata import open_wsi

from lazyslide._utils import find_stack_level, warn_deprecated

_OFFLINE_ENV_VARS = ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE")
_TRUE_VALUES = {"1", "ON", "TRUE", "YES"}
_DATASET_REVISION_ENV = "LAZYSLIDE_DATASET_REVISION"


def _hf_offline() -> bool:
    return any(
        os.environ.get(name, "").strip().upper() in _TRUE_VALUES
        for name in _OFFLINE_ENV_VARS
    )


def _download_dataset_file(repo_id: str, filename: str, revision: str | None) -> str:
    return hf_hub_download(
        repo_id,
        filename,
        repo_type="dataset",
        revision=revision,
        local_files_only=_hf_offline(),
    )


def _dataset_revision(repo_id: str) -> str | None:
    """Resolve the dataset revision, honoring an explicit immutable CI pin."""
    revision = os.environ.get(_DATASET_REVISION_ENV, "").strip()
    if revision:
        return revision

    from packaging.version import Version

    from lazyslide import __version__

    version = Version(__version__)
    revision = f"v{version.base_version}"

    if _hf_offline():
        if version.public != version.base_version or version.local is not None:
            return None
        return revision

    api = HfApi()
    refs = api.list_repo_refs(repo_id, repo_type="dataset")
    tags = [tag.name for tag in refs.tags]
    return revision if revision in tags else None


def _load_dataset(slide_file, zarr_file, with_data=True, pbar=None):
    if pbar is not None:
        warn_deprecated(
            "`pbar` of the `zs.datasets` loaders is deprecated since v0.8.0 and will "
            "be removed in v0.14.0; it has no effect, remove it."
        )
    REPO_ID = "RendeiroLab/LazySlide-data"
    revision = _dataset_revision(REPO_ID)

    slide = _download_dataset_file(REPO_ID, slide_file, revision=revision)
    slide_zarr = None
    if with_data:
        slide_zarr_zip = Path(
            _download_dataset_file(REPO_ID, zarr_file, revision=revision)
        )
        # Not <stem>.zarr next to the slide: open_wsi(slide) attaches that store by
        # default, and a write() or rmtree of it would change the dataset
        slide_zarr = slide_zarr_zip.parent / "precomputed" / slide_zarr_zip.stem
        # Unzip the zarr file if it is a zip file
        # But only if it is not already unzipped
        if not slide_zarr.exists():
            from tempfile import TemporaryDirectory
            from zipfile import ZipFile

            slide_zarr.parent.mkdir(exist_ok=True)
            # Extract elsewhere and move the store in at once: an interrupted
            # extraction leaves no partial store, and no other process opens one.
            # tmpXXXXXXXX in the snapshot is as long as precomputed, so no path
            # gets longer than the store's own (Windows MAX_PATH)
            with TemporaryDirectory(dir=slide_zarr_zip.parent) as tmp:
                with ZipFile(slide_zarr_zip, "r") as zip_ref:
                    zip_ref.extractall(tmp)
                try:
                    Path(tmp, slide_zarr.name).rename(slide_zarr)
                except OSError:
                    # Another process moved its store in first; use that one
                    if not slide_zarr.exists():
                        raise
            old_zarr = slide_zarr_zip.with_suffix("")
            if old_zarr.exists():
                warnings.warn(
                    f"The store of this dataset is now extracted to {slide_zarr}. "
                    f"{old_zarr}, where earlier versions of LazySlide extracted it, "
                    "is left as it is, and open_wsi() of the slide attaches it by "
                    "default. Delete it if it holds nothing you need.",
                    stacklevel=find_stack_level(),
                )
    return open_wsi(slide, store=str(slide_zarr) if with_data else None)


def sample(with_data: bool = True, pbar: bool | None = None):
    """
    Load a small sample slide (~1.9 MB).

    Source: https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/CMU-1-Small-Region.svs

    Parameters
    ----------
    with_data : bool, default: True
        Whether to load the associated zarr storage data.
    pbar : bool, optional
        .. deprecated:: 0.8.0
            Has no effect and will be removed in 0.14.0.

    Returns
    -------
    :class:`WSIData <wsidata.WSIData>`
        The loaded whole-slide image object.

    """
    return _load_dataset(
        "sample.svs", "sample.zarr.zip", with_data=with_data, pbar=pbar
    )


def gtex_artery(with_data: bool = True, pbar: bool | None = None):
    """
    A GTEX artery slide.

    Source: https://gtexportal.org/home/histologyPage, GTEX-1117F-0526

    Parameters
    ----------
    with_data : bool, default: True
        Whether to load the associated zarr storage data.
    pbar : bool, optional
        .. deprecated:: 0.8.0
            Has no effect and will be removed in 0.14.0.

    Returns
    -------
    :class:`WSIData <wsidata.WSIData>`
        The loaded whole-slide image object.

    """
    return _load_dataset(
        "GTEX-1117F-0526.svs",
        "GTEX-1117F-0526.zarr.zip",
        with_data=with_data,
        pbar=pbar,
    )


def gtex_small_intestine(with_data: bool = True, pbar: bool | None = None):
    """
    A small GTEX small intestine slide for testing purposes.

    Source: https://gtexportal.org/home/histologyPage, GTEX-11DXX-1626.svs

    Parameters
    ----------
    with_data : bool, default: True
        Whether to load the associated zarr storage data.
    pbar : bool, optional
        .. deprecated:: 0.8.0
            Has no effect and will be removed in 0.14.0.

    Returns
    -------
    :class:`WSIData <wsidata.WSIData>`
        The loaded whole-slide image object.

    """
    return _load_dataset(
        "GTEX-11DXX-1626.svs",
        "GTEX-11DXX-1626.zarr.zip",
        with_data=with_data,
        pbar=pbar,
    )


def lung_carcinoma(with_data: bool = True, pbar: bool | None = None):
    """
    A lung carcinoma slide.

    Source: https://idr.openmicroscopy.org/webclient/img_detail/9846318/?dataset=10801

    Parameters
    ----------
    with_data : bool, default: True
        Whether to load the associated zarr storage data.
    pbar : bool, optional
        .. deprecated:: 0.8.0
            Has no effect and will be removed in 0.14.0.

    Returns
    -------
    :class:`WSIData <wsidata.WSIData>`
        The loaded whole-slide image object.

    """

    return _load_dataset(
        "lung_carcinoma.ndpi", "lung_carcinoma.zarr.zip", with_data=with_data, pbar=pbar
    )
