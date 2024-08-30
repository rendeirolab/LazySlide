import warnings
from pathlib import Path
from typing import Optional

import pandas as pd
from rich import print
from typer import Typer, Argument, Option

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


@app.command()
def preprocess(
    wsi: str = WSI,
    min_tissue_area: float = 1e-3,
    min_hole_area: float = 1e-5,
    detect_holes: bool = True,
    filter_artifacts: bool = True,
    tissue_qc: str = "brightness,redness",
    filter_tissue: bool = True,
    tile_px: int = Argument(..., help="The size of the tile in pixels"),
    stride_px: int = Option(
        None, help="The stride of the tile in pixels, by default equal to tile_px"
    ),
    mpp: float = Option(None, help="The microns per pixel"),
    tile_qc: str = "focus,contrast",
    filter_tiles: bool = True,
    report: bool = False,
    output: Optional[str] = OUTPUT,
):
    import lazyslide as zs

    print(f"Read slide file {wsi}")
    wsi = zs.open_wsi(wsi, backed_file=output)
    zs.pp.find_tissue(
        wsi,
        min_tissue_area=min_tissue_area,
        min_hole_area=min_hole_area,
        detect_holes=detect_holes,
        filter_artifacts=filter_artifacts,
    )

    # zs.pp.tissue_qc(wsi, tissue_qc.split(","))
    # if filter_tissue:
    #     tissue_tb = wsi.sdata["tissues"]
    #     wsi.sdata["tissues"] = tissue_tb[tissue_tb["qc"]]

    zs.pp.tiles(wsi, tile_px=tile_px, stride_px=stride_px, mpp=mpp)

    # zs.pp.tiles_qc(wsi, tile_qc.split(","))
    # if filter_tiles:
    #     tile_tb = wsi.sdata["tiles"]
    #     wsi.sdata["tiles"] = tile_tb[tile_tb["qc"]]
    #
    # if report:
    #     zs.pl.qc_summary(wsi, ["brightness", "redness"], ["focus", "contrast"])

    wsi.save()
    print(f"Saved to {wsi.sdata.path}")


@app.command()
def feature(
    slide: str = WSI,
    model: str = Argument(..., help="A model name or the path to the model file"),
    slide_encoder: str = None,
    output: Optional[str] = OUTPUT,
):
    import lazyslide as zs

    wsi = zs.open_wsi(slide, backed_file=output)
    print(f"Read slide file {slide}")
    print(f"Extract features using model {model}")
    zs.tl.feature_extraction(wsi, model, slide_encoder=slide_encoder)
    wsi.save()
    print(f"Write to {wsi.sdata.path}")


@app.command()
def agg_wsi(
    slide_table: Path = Argument(..., help="The slide table file"),
    output: Optional[str] = OUTPUT,
):
    from wsi_data import agg_wsi

    print(f"Read slide table {slide_table}")
    slides_table = pd.read_csv(slide_table)
    data = agg_wsi(slides_table, "features")
    data.write_zarr(output)
