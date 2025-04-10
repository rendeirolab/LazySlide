import pooch
from wsidata import open_wsi

ENTRY = pooch.create(
    path=pooch.os_cache("lazyslide"),
    base_url="https://lazyslide.blob.core.windows.net/lazyslide-data",
    registry={
        "sample.svs": "sha256:ed92d5a9f2e86df67640d6f92ce3e231419ce127131697fbbce42ad5e002c8a7",
        "sample.zarr.zip": "sha256:075a3ab61e6958673d79612cc29796a92cf875ad049fc1fe5780587968635378",
        "GTEX-1117F-0526.svs": "sha256:222ab7f2bb42dcd0bcfaccd910cb13be452b453499e6117ab553aa6cd60a135e",
        "GTEX-1117F-0526.zarr.zip": "sha256:2323b656322d2dcc7e9d18aaf586b39a88bf8f2a3959f642f109eb54268f3732",
        "lung_carcinoma.ndpi": "sha256:3297b0a564f22940208c61caaca56d97ba81c9b6b7816ebc4042a087e557f85e",
        "lung_carcinoma.zarr.zip": "sha256:0a8ccfc608f55624b473c6711b55739c3279d3b6fc5b654395dfc23b010bf866",
    },
)

logger = pooch.get_logger()
logger.setLevel("WARNING")


def _load_dataset(slide_file, zarr_file, with_data=True, pbar=False):
    slide = ENTRY.fetch(slide_file)
    _ = ENTRY.fetch(
        zarr_file,
        progressbar=pbar,
        processor=pooch.Unzip(extract_dir=zarr_file.rstrip(".zip")),
    )
    store = "auto" if with_data else None
    return open_wsi(slide, store=store)


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
