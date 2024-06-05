from typer import Typer
from lazyslide import WSI

import warnings
import logging
from rich.logging import RichHandler

FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

log = logging.getLogger("LAZYSLIDE")
log.setLevel(logging.INFO)
warnings.filterwarnings("ignore", category=UserWarning)

app = Typer(pretty_exceptions_show_locals=False)


@app.command()
def tissue(slide: str):
    from lazyslide.pp import find_tissue

    log.info(f"Read slide file {slide}")
    wsi = WSI(slide)
    find_tissue(wsi)
    wsi.write()
    log.info(f"Write to {wsi.file}")


@app.command()
def tile(slide: str, tile_px: int, mpp: float):
    from lazyslide.pp import tiles

    log.info(f"Read slide file {slide}")
    wsi = WSI(slide)
    tiles(wsi, tile_px=tile_px, mpp=mpp)
    wsi.write()
    log.info(f"Write to {wsi.file}")
