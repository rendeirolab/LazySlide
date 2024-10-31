import warnings
from pathlib import Path
from typing import Optional

import typer
from typer import Typer, Argument, Option
from rich import print


warnings.filterwarnings("ignore", category=UserWarning)

app = Typer(pretty_exceptions_show_locals=False)

# Define shared parameters
WSI = Argument(..., help="The whole slide image file to process")
OUTPUT = Option(
    None,
    "--output",
    "-o",
    help="By default will write to a zarr file with the same name as the slide file",
)
REPORT_OUTPUT = Option(
    ".",
    "--output",
    "-o",
    help="By default will write to running directory with the same name as the slide file",
)
TISSUE_QC = Option(
    "brightness,redness",
    "--tissue-qc",
    help="The quality control metrics for tissue detection, multiple metrics separated by comma",
)
TILE_QC = Option(
    "focus,contrast",
    "--tile-qc",
    help="The quality control metrics for tile detection, multiple metrics separated by comma",
)
TILE_PX = Option(512, "--tile-px", help="The size of the tile in pixels")
STRIDE_PX = Option(
    None,
    "--stride-px",
    help="The stride of the tile in pixels, by default equal to tile_px",
)


@app.command()
def info(slide: str = WSI):
    """Quick inspect the properties of a slide"""
    from wsidata import open_wsi
    from rich.console import Console
    from rich.table import Table

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


@app.command()
def preprocess(
    wsi: str = WSI,
    min_tissue_area: float = 1e-3,
    min_hole_area: float = 1e-5,
    detect_holes: bool = True,
    qc: bool = True,
    filter_artifacts: bool = True,
    tissue_qc: str = TISSUE_QC,
    filter_tissue: bool = True,
    tile_px: int = TILE_PX,
    stride_px: int = STRIDE_PX,
    mpp: float = Option(None, help="The microns per pixel"),
    slide_mpp: float = Option(None, help="The microns per pixel of the slide"),
    tile_qc: str = TILE_QC,
    filter_tiles: bool = True,
    output: Optional[Path] = OUTPUT,
):
    import lazyslide as zs
    from lazyslide._const import Key

    print(f"Read slide file {wsi}")
    wsi = zs.open_wsi(wsi, backed_file=output)
    zs.pp.find_tissues(
        wsi,
        min_tissue_area=min_tissue_area,
        min_hole_area=min_hole_area,
        detect_holes=detect_holes,
        filter_artifacts=filter_artifacts,
    )

    if qc:
        if tissue_qc is not None:
            tissue_metrics = tissue_qc.split(",")
            zs.pp.tissues_qc(wsi, tissue_metrics)
            if filter_tissue:
                tissue_tb = wsi[Key.tissue]
                wsi[Key.tissue] = tissue_tb[tissue_tb[Key.tissue_qc]]

    zs.pp.tile_tissues(
        wsi, tile_px=tile_px, stride_px=stride_px, mpp=mpp, slide_mpp=slide_mpp
    )

    if qc:
        if tile_qc is not None:
            tile_metrics = tile_qc.split(",")
            zs.pp.tiles_qc(wsi, tile_metrics)
            if filter_tiles:
                tile_tb = wsi[Key.tiles]
                wsi[Key.tiles] = tile_tb[tile_tb[Key.tile_qc]]

    wsi.save()
    print(f"Saved to {wsi.path}")


@app.command()
def report(
    slide: str = WSI,
    tissue_qc: str = TISSUE_QC,
    tile_qc: str = TILE_QC,
    output: Optional[Path] = REPORT_OUTPUT,
):
    from wsidata import open_wsi
    import lazyslide as zs

    wsi = open_wsi(slide)
    print(f"Read slide file {wsi}")
    report_fig = zs.pl.qc_summary(wsi, tissue_qc.split(","), tile_qc.split(","))
    if output.is_dir():
        output = output / wsi.backed_file.with_suffix(".pdf").name
    report_fig.savefig(output)
    print(f"Saved to {output}")


@app.command()
def feature(
    slide: str = WSI,
    model: str = Argument(..., help="A model name or the path to the model file"),
    slide_encoder: str = "mean",
    device: str = "cpu",
    output: Optional[str] = OUTPUT,
):
    import lazyslide as zs

    wsi = zs.open_wsi(slide, backed_file=output)
    print(f"Read slide file {slide}")
    print(f"Extract features using model {model}")
    zs.tl.feature_extraction(wsi, model, slide_agg=slide_encoder, device=device)
    wsi.save()
    print(f"Write to {wsi.path}")


@app.command()
def agg_wsi(
    slide_table: Path = Argument(..., help="The slide table file"),
    output: Optional[str] = OUTPUT,
):
    from wsidata import agg_wsi
    import pandas as pd

    print(f"Read slide table {slide_table}")
    slides_table = pd.read_csv(slide_table)
    data = agg_wsi(slides_table, "features")
    data.write_zarr(output)


typer_click_object = typer.main.get_command(app)
