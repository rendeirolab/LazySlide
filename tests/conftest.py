from pathlib import Path

import os
import platform

import pytest


OPENSLIDE_DOWNLOAD_URL = (
    "https://github.com/openslide/openslide-bin/releases/download/"
    "v20231011/openslide-win64-20231011.zip"
)
LIBVIPS_DOWNLOAD_URL = (
    "https://github.com/libvips/build-win64-mxe/releases/download/"
    "v8.15.1/vips-dev-w64-all-8.15.1.zip"
)


def download_file(url, target):
    import requests
    import zipfile
    import io

    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(target)
    return z.namelist()[0]


@pytest.fixture(scope="session", autouse=True)
def load_module_windows():
    if platform.system() == "Windows":
        target = Path(__file__).parent / "lib"
        target.mkdir(exist_ok=True)

        if not Path(target / "openslide").exists():
            openslide_folder = download_file(OPENSLIDE_DOWNLOAD_URL, target)
            os.rename(target / openslide_folder, target / "openslide")

        if not Path(target / "vips").exists():
            vips_folder = download_file(LIBVIPS_DOWNLOAD_URL, target)
            os.rename(target / vips_folder, target / "vips")


def import_windows_modules():
    if platform.system() == "Windows":
        target = Path(__file__).parent / "lib"
        print(target)
        with os.add_dll_directory(str(target / "openslide" / "bin")):
            import openslide

        os.environ["PATH"] = str(target / "vips" / "bin") + ";" + os.environ["PATH"]
        import pyvips
