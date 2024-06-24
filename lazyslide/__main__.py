import warnings
from typing import Optional

from rich import print
from typer import Typer, Argument, Option

from lazyslide import WSI

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
    from lazyslide.pp import find_tissue

    print(f"Read slide file {slide}")
    wsi = WSI(slide)
    find_tissue(wsi)
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
    from lazyslide.pp import tiles

    print(f"Read slide file {slide}")
    wsi = WSI(slide)
    tiles(wsi, tile_px=tile_px, stride_px=stride_px, mpp=mpp)
    wsi.write(output)
    print(f"Write to {wsi.file}")
