__all__ = ["reader_datatree"]

from dataclasses import asdict

from xarray import open_zarr, DataArray
from datatree import DataTree
from spatialdata.models import Image2DModel
from spatialdata.transformations import Identity, Scale

from ._store import create_reader_store


def reader_datatree(
    reader,
    chunks=(3, 512, 512),
) -> DataTree:
    store = create_reader_store(reader)
    img_dataset = open_zarr(store, consolidated=False, mask_and_scale=False)

    images = {}
    for level, key in enumerate(list(img_dataset.keys())):
        suffix = key if key != "0" else ""

        scale_image = DataArray(
            img_dataset[key].transpose("S", f"Y{suffix}", f"X{suffix}"),
            dims=("c", "y", "x"),
        ).chunk(chunks)

        scale_factor = reader.properties.level_downsample[level]

        if scale_factor == 1:
            transform = Identity()
        else:
            transform = Scale([scale_factor, scale_factor], axes=("y", "x"))

        scale_image = Image2DModel.parse(
            scale_image[:3, :, :],
            transformations={"global": transform},
            c_coords=("r", "g", "b"),
        )
        scale_image.coords["y"] = scale_factor * scale_image.coords["y"]
        scale_image.coords["x"] = scale_factor * scale_image.coords["x"]

        images[f"scale{key}"] = scale_image

    slide_image = DataTree.from_dict(images)
    slide_image.attrs = asdict(reader.properties)
    return slide_image
