# %%
import os
import tempfile
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download, upload_file
from wsidata import open_wsi

import lazyslide as zs


def download_file(filename):
    return hf_hub_download(
        repo_id="RendeiroLab/LazySlide-data",
        filename=filename,
        repo_type="dataset",
    )


def should_exclude_file(file_path):
    """Check if a file should be excluded from the zip (platform-specific files)."""
    file_path = Path(file_path)

    # Exclude macOS specific files
    if file_path.name.startswith("._") or file_path.name == ".DS_Store":
        return True

    # Exclude __MACOSX directory and its contents
    if "__MACOSX" in file_path.parts:
        return True

    # Exclude other common platform-specific files
    if file_path.name in [".Thumbs.db", "desktop.ini", ".directory"]:
        return True

    return False


def create_filtered_zip(zarr_path, zip_path):
    """Create a zip file from zarr directory, excluding platform-specific files."""
    print(f"Creating zip file: {zip_path}")

    with zipfile.ZipFile(zip_path, "w") as zipf:
        # Walk through the zarr directory and add all files to zip
        for root, dirs, files in os.walk(zarr_path):
            # Filter out platform-specific directories
            dirs[:] = [d for d in dirs if not should_exclude_file(Path(root) / d)]

            for file in files:
                file_path = Path(root) / file

                # Skip platform-specific files
                if should_exclude_file(file_path):
                    print(f"Excluding: {file_path}")
                    continue

                # Create an archive path relative to the zarr directory
                arcname = file_path.relative_to(zarr_path.parent)
                zipf.write(file_path, arcname)
                print(f"Added: {arcname}")

    print(f"Zip file created successfully: {zip_path}")
    return zip_path


def create_zip(
    wsi,
    name,
    output_path=None,
):
    # Create zarr and zip
    zarr_path = output_path / f"{name}.zarr"
    zip_path = output_path / f"{name}.zarr.zip"

    print("Writing zarr file...")
    wsi.write(zarr_path)

    # Create filtered zip
    create_filtered_zip(zarr_path, zip_path)

    print(f"Saved to: {zip_path}")


def process_sample(wsi_path, geojson_path=None):
    print("Processing whole slide image...")
    wsi = open_wsi(wsi_path, store=None)
    zs.pp.find_tissues(wsi)
    zs.seg.tissue(wsi, key_added="dl-tissue")
    zs.pp.tile_tissues(wsi, 256)
    zs.tl.feature_extraction(wsi, "resnet50")
    if geojson_path:
        zs.io.load_annotations(wsi, geojson_path)
    return wsi


def process_gtex_artery(svs_path, geojson_path=None):
    print("Processing GTEx artery slide...")
    wsi = open_wsi(svs_path, store=None)
    zs.pp.find_tissues(wsi)
    zs.seg.tissue(wsi, key_added="dl-tissue")
    zs.pp.tile_tissues(wsi, 256, mpp=0.5)
    zs.tl.feature_extraction(wsi, "resnet50")
    zs.tl.feature_extraction(wsi, "uni2")
    if geojson_path:
        zs.io.load_annotations(wsi, geojson_path)
    return wsi


def process_lung_carcinoma(svs_path, geojson_path=None):
    print("Processing lung carcinoma slide...")
    wsi = open_wsi(svs_path, store=None)
    zs.pp.find_tissues(wsi)
    zs.pp.tile_tissues(wsi, 256)
    zs.tl.feature_extraction(wsi, "virchow", pbar=True)
    if geojson_path:
        zs.io.load_annotations(wsi, geojson_path)
    return wsi


def process_gtex_small_intestine(svs_path, geojson_path=None):
    print("Processing sample GTEx small intestine slide...")
    wsi = open_wsi(svs_path, store=None)
    zs.pp.find_tissues(wsi)
    zs.pp.tile_tissues(wsi, 128)
    zs.tl.feature_extraction(wsi, "plip")
    texts = ["mucosa", "submucosa", "musclaris", "lymphocyte"]
    text_embeddings = zs.tl.text_embedding(texts, model="plip")
    zs.tl.text_image_similarity(wsi, text_embeddings, model="plip", softmax=True)
    return wsi


def test_load_dataset(slide_file, zarr_zip):
    slide_zarr = Path(str(zarr_zip).replace(".zip", ""))
    with zipfile.ZipFile(zarr_zip, "r") as zip_ref:
        zip_ref.extractall(slide_zarr.parent)
    return open_wsi(slide_file, store=str(slide_zarr))


# %%
root = Path(__file__).parent.parent
output_path = root / Path("generated_zarr_zip")
output_path.mkdir(parents=True, exist_ok=True)
repo_id = "RendeiroLab/LazySlide-data"

# %%
REGISTRY = [
    ("sample.svs", "sample.geojson", process_sample),
    ("GTEX-1117F-0526.svs", "GTEX-1117F-0526.geojson", process_gtex_artery),
    ("GTEX-11DXX-1626.svs", None, process_gtex_small_intestine),
    ("lung_carcinoma.ndpi", "lung_carcinoma.geojson", process_lung_carcinoma),
]

for slide, geojson, process_func in REGISTRY:
    print(f"Processing {slide}...")
    svs_path = download_file(slide)
    if geojson:
        geojson_path = download_file(geojson)
    else:
        geojson_path = None
    slide_path = Path(output_path / slide)
    zip_path = slide_path.with_suffix(".zarr.zip")
    wsi = process_func(svs_path, geojson_path)
    create_zip(wsi, slide_path.stem, output_path=output_path)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Extract all to tmp_dir and list
        with zipfile.ZipFile(zip_path, "r") as zipf:
            zipf.extractall(Path(tmp_dir))
        files = os.listdir(tmp_dir)
        print(files)

    test_load_dataset(svs_path, zip_path)
