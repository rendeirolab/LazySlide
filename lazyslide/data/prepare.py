from __future__ import annotations

import json
from pathlib import Path
from typing import List, Literal

from PIL.Image import fromarray

import pandas as pd
from fsspec.core import open
from rich.progress import track

from lazyslide.wsi import WSI
from lazyslide.get import tile_images, n_tiles

DATA_TYPES = Literal[
    "tile_images",
    "tissue_images",
    "cell_images",
    "tile_features",
    "tissue_features",
    "cell_features",
    "graph",
]


# TODO: The feature should be saved using safetensors


class DiskDatasetBuilder:
    def __init__(self, wsi: WSI, output_dir: str):
        self.wsi = wsi
        self.output_dir = Path(output_dir)
        self.slide_dir = self._prepare_slide_dir()

    def _prepare_slide_dir(self):
        slide_dir = self.output_dir / self.wsi.name
        slide_dir.mkdir(parents=True, exist_ok=True)
        metadata_file = slide_dir / "metadata.json"
        if not metadata_file.exists():
            with open(metadata_file, "w") as f:
                json.dump(self.wsi.metadata.model_dump(), f)
        with open(slide_dir / "annotation.json", "w") as f:
            json.dump(self.wsi.get_slide_annotations(), f)
        return slide_dir

    def write_tile_images(
        self,
        tile_key: str = "tiles",
        raw=False,
        color_norm: str = None,
        sample_n: int = None,
        format: str = "jpg",
        pbar: bool = True,
    ):
        """Write tile images to disk

        This will save to the output_dir/"{slide_name}"/"tile_images"/tile_key directory.

        """
        tile_dir = self.slide_dir / "tile_images" / tile_key
        tile_dir.mkdir(parents=True, exist_ok=True)
        tile_data = []

        def img_name(tile, format):
            return f"x={tile.x},y={tile.y},id={tile.id},tissue_id={tile.tissue_id}.{format}"

        for tile in track(
            tile_images(
                self.wsi, tile_key, raw=raw, color_norm=color_norm, sample_n=sample_n
            ),
            total=n_tiles(self.wsi, tile_key),
            disable=not pbar,
            description="Writing tile images",
        ):
            tile_image = tile.image
            tile_name = img_name(tile, format)
            fromarray(tile_image).save(tile_dir / tile_name)
            tile_data.append([tile.x, tile.y, tile.id, tile.tissue_id, tile_name])
        # Export tile spec
        with open(tile_dir / "tile_spec.json", "w") as f:
            json.dump(self.wsi.get_tile_spec(tile_key).model_dump(), f)
        # Export tile annotations
        columns = ["x", "y", "id", "tissue_id", "filename"]
        tile_data = pd.DataFrame(tile_data, columns=columns)
        tile_annotations = self.wsi.sdata.points[tile_key].compute().set_index("id")
        tile_annotations = tile_annotations.loc[tile_data["id"]]
        for key in tile_annotations:
            if key not in columns:
                tile_data[key] = tile_annotations[key]
        tile_data.to_csv(tile_dir / "tiles.csv", index=False)
