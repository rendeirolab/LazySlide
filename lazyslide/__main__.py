import warnings
from typing import Optional

from rich import print
from typer import Typer, Argument, Option

warnings.filterwarnings("ignore", category=UserWarning)

app = Typer(pretty_exceptions_show_locals=False)

# Define shared parameters
SLIDE = Argument(..., help="The slide file to process")
OUTPUT = Option(
    None,
    "--output",
    "-o",
    help="By default will write to a zarr file with the same name as the slide file",
)


@app.command()
def tissue(
    slide: str = SLIDE,
    output: Optional[str] = OUTPUT,
):
    import lazyslide as zs

    print(f"Read slide file {slide}")
    wsi = zs.WSI(slide)
    zs.pl.find_tissue(wsi)
    wsi.write(output)
    print(f"Write to {wsi.file}")


@app.command()
def tile(
    slide: str = SLIDE,
    tile_px: int = Argument(..., help="The size of the tile in pixels"),
    stride_px: int = Option(
        None, help="The stride of the tile in pixels, by default equal to tile_px"
    ),
    mpp: float = Option(None, help="The microns per pixel"),
    output: Optional[str] = OUTPUT,
):
    import lazyslide as zs

    print(f"Read slide file {slide}")
    wsi = zs.WSI(slide)
    zs.tiles(wsi, tile_px=tile_px, stride_px=stride_px, mpp=mpp)
    wsi.write(output)
    print(f"Write to {wsi.file}")


@app.command()
def feature(
    slide: str = SLIDE,
    model: str = Argument(..., help="A model name or the path to the model file"),
    output: Optional[str] = OUTPUT,
):
    import lazyslide as zs

    print(f"Read slide file {slide}")
    wsi = zs.WSI(slide)
    zs.feature_extraction(wsi, model)
    wsi.write(output)
    print(f"Write to {wsi.file}")


@app.command()
def list_models(
    pattern: str = Option(
        None, "--pattern", "-p", help="A pattern to filter the list of models"
    ),
):
    """
    List the available models from timm.
    """
    import re
    import timm

    models = timm.list_models(pattern)
    # A rich table for the list of models
    for model in models:
        print(model)
