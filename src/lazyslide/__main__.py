from __future__ import annotations

import warnings
from pathlib import Path
from typing import Annotated, Optional

from cyclopts import App, Parameter, validators
from rich import print

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

app = App(
    name="lazyslide",
    help="Whole slide image processing",
)


@app.command
def info(slide: Annotated[Path, Parameter(validator=validators.Path(exists=True))]):
    """Quick inspect the properties of a slide

    Parameters
    ----------
    slide : str
        The whole slide image file to process

    """
    from rich.console import Console
    from rich.table import Table
    from wsidata import open_wsi

    wsi = open_wsi(slide)

    c = Console()

    c.print(f"Slide file: [italic red]{slide}[/italic red]")
    c.print(f"Reader: [bold purple]{wsi.reader.name}[/bold purple]")
    c.print(
        f"Physical pixel size (Microns per pixels): [bold cyan]{wsi.properties.mpp}[/bold cyan] µm/px"
    )
    c.print(
        f"Magnification: [bold cyan]{int(wsi.properties.magnification)}X[/bold cyan]"
    )

    t = Table(title="Pyramids")
    t.add_column("Level", style="cyan")
    t.add_column("Height", style="magenta")
    t.add_column("Width", style="green")
    t.add_column("Downsample", style="yellow")

    for i, (h, w, d) in enumerate(wsi.fetch.pyramids().values):
        t.add_row(str(i), str(int(h)), str(int(w)), str(d))

    c.print(t)


@app.command
def preprocess(
    wsi: str,
    tile_px: int,
    stride_px: int | None = None,
    min_tissue_area: float = 1e-3,
    min_hole_area: float = 1e-5,
    detect_holes: bool = True,
    filter_artifacts: bool = True,
    mpp: float | None = None,
    slide_mpp: float | None = None,
    output: Path | None = None,
):
    """
    Preprocess a whole slide image

    Steps:
        Find tissues -> Tile tissues

    Parameters
    ----------
    wsi : str
        The whole slide image file to process
    tile_px : int
        The size of the tile in pixels
    stride_px : int
        The stride of the tile in pixels
    min_tissue_area : float
        The minimum tissue area to consider as tissue
    min_hole_area : float
        The minimum hole area to consider as hole
    detect_holes : bool
        Whether to detect holes in the tissue
    filter_artifacts : bool
        Whether to filter artifacts
    mpp : float
        The microns per pixel
    slide_mpp : float
        The microns per pixel of the slide
    output : Path
        The output path to save the processed slide

    """
    import lazyslide as zs

    print(f"Read slide file {wsi}")
    wsi = zs.open_wsi(wsi, backed_file=output)
    zs.pp.find_tissues(
        wsi,
        min_tissue_area=min_tissue_area,
        min_hole_area=min_hole_area,
        detect_holes=detect_holes,
        filter_artifacts=filter_artifacts,
    )

    zs.pp.tile_tissues(
        wsi, tile_px=tile_px, stride_px=stride_px, mpp=mpp, slide_mpp=slide_mpp
    )

    wsi.write(overwrite=True)
    print(f"Saved to {wsi.path}")


@app.command
def qc(
    wsi: str,
    tissue_qc: str = "brightness,redness",
    filter_tissue: bool = False,
    tile_qc: str = "focus,contrast",
    filter_tiles: bool = True,
):
    """
    Quality control of a whole slide image
    """
    import lazyslide as zs
    from lazyslide._const import Key

    wsi = zs.open_wsi(wsi)

    if tissue_qc is not None:
        tissue_metrics = tissue_qc.split(",")
        zs.pp.tissues_qc(wsi, tissue_metrics)
        if filter_tissue:
            tissue_tb = wsi[Key.tissue]
            wsi[Key.tissue] = tissue_tb[tissue_tb[Key.tissue_qc]]
    if tile_qc is not None:
        tile_metrics = tile_qc.split(",")
        zs.pp.tiles_qc(wsi, tile_metrics)
        if filter_tiles:
            tile_tb = wsi[Key.tiles]
            wsi[Key.tiles] = tile_tb[tile_tb[Key.tile_qc]]

    wsi.write(overwrite=True)
    print(f"Saved to {wsi.path}")


# @app.command
# def report(
#     slide: str = WSI,
#     tissue_qc: str = TISSUE_QC,
#     tile_qc: str = TILE_QC,
#     output: Optional[Path] = REPORT_OUTPUT,
# ):
#     from wsidata import open_wsi
#     import lazyslide as zs
#
#     wsi = open_wsi(slide)
#     print(f"Read slide file {wsi}")
#     report_fig = zs.pl.qc_summary(wsi, tissue_qc.split(","), tile_qc.split(","))
#     if output.is_dir():
#         output = output / wsi.backed_file.with_suffix(".pdf").name
#     report_fig.savefig(output)
#     print(f"Saved to {output}")
#
#
@app.command
def feature(
    slide: str,
    model: str,
    slide_agg: str = "mean",
    device: str = None,
    num_workers: int | str = "auto",
    output: Optional[str] = None,
):
    """
    Extract features from a whole slide image

    Parameters
    ----------
    slide : str
        The whole slide image file to process
    model : str
        A model name or the path to the model file
    slide_agg : str
        The slide aggregation method
    device : str
        The device to run the model
    num_workers : int
        The number of workers to use
    output : str
        The output path to save the processed slide

    """
    import lazyslide as zs

    wsi = zs.open_wsi(slide, backed_file=output)
    print(f"Read slide file {slide}")
    print(f"Extract features using model {model}")

    if Path(model).exists():
        model_name = "user_model"
    else:
        model_name = model

    key_added = f"{model_name}"

    zs.tl.feature_extraction(
        wsi, model, device=device, num_workers=num_workers, key_added=key_added
    )
    zs.tl.feature_aggregation(wsi, feature_key=key_added, encoder=slide_agg)
    wsi.write_element(key_added, overwrite=True)
    print(f"Write to {wsi.path}")


@app.command
def agg(
    slide_table: Path,
    feature_key: str = "features",
    agg_key: str = "agg_slide",
    wsi_col: str | None = None,
    store_col: str | None = None,
    output: Path | None = None,
):
    """
    Aggregate features from a slide table

    Parameters
    ----------
    slide_table : Path
        The slide table file
    output : str
        The output path to save the aggregated features

    """
    import pandas as pd
    from wsidata import agg_wsi

    print(f"Read slide table {slide_table}")
    slides_df = pd.read_csv(slide_table)
    root = Path(slide_table).parent
    _assure_path(slides_df, wsi_col, root)
    _assure_path(slides_df, store_col, root)

    data = agg_wsi(
        slides_df,
        feature_key=feature_key,
        agg_key=agg_key,
        wsi_col=wsi_col,
        store_col=store_col,
    )
    if output is None:
        output = Path(f"./agg_{feature_key}.zarr")

    if not output.name.endswith(".zarr"):
        if not is_zarr_dir(output):
            if not is_dir_empty(output):
                raise FileExistsError(f"Output directory {output} is not empty")

    data.write_zarr(output)


def _assure_path(slides_df, col=None, root=None):
    if col is not None:
        wsis = slides_df[col]
        # get the first one
        wsi = wsis[0]
        origin_path = Path(wsi)
        prefix_path = root / origin_path
        if not origin_path.exists():
            # if the file in the table is not existed,
            # try to find it in the same directory as the table
            if prefix_path.exists():
                slides_df[col] = [str(root / Path(wsi)) for wsi in wsis]
            else:
                raise FileNotFoundError(f"File {origin_path} not existed.")


def is_zarr_dir(path: Path):
    """
    Detect if the given directory is a Zarr storage using the Zarr library.

    Parameters:
        path (str): The path to the directory.

    Returns:
        bool: True if the directory is a Zarr storage, False otherwise.
    """
    import zarr

    try:
        zarr.open_group(path, mode="r")
        return True
    except Exception:
        return False


def is_dir_empty(path: Path):
    """
    Check if a directory is empty

    Parameters:
        path (str): The path to the directory.

    Returns:
        bool: True if the directory is empty, False otherwise.
    """
    return len(list(path.iterdir())) == 0
