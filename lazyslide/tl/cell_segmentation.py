from typing import Literal

from lazyslide.wsi import WSI

# TODO: There should be two ways of running cell segmentation:
#       1. Launch at Python main threads.
#       2. Prepare a file directory for the cell segmentation.
#          And then, run the cell segmentation by docker.


def cell_segmentation(
    wsi: WSI,
    method: Literal["cellpose", "stardist", "deepcell", "cellvit"] = "cellpose",
):
    pass
