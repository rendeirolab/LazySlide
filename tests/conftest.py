from pathlib import Path

import os
import platform

import pytest


OPENSLIDE_DOWNLOAD_URL = (
    "https://github.com/openslide/openslide-bin/releases/download/"
    "v20231011/openslide-win64-20231011.zip"
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


def import_windows_modules():
    if platform.system() == "Windows":
        target = Path(__file__).parent / "lib"
        print(target)
        with os.add_dll_directory(str(target / "openslide" / "bin")):
            import openslide


@pytest.fixture(scope="session", autouse=True)
def test_slide():
    return Path(__file__).parent / "data" / "CMU-1-Small-Region.svs"
