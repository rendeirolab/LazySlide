import geopandas as gpd
import numpy as np
from shapely.affinity import scale, translate
from torch.utils.data import DataLoader
from wsidata import WSIData
from wsidata.io import add_shapes

from lazyslide.cv import PolygonMerger
from lazyslide.models.base import SegmentationModel


def semantic_segmentation(
    wsi: WSIData,
    model: SegmentationModel,
    tile_key="tiles",
    transform=None,
    batch_size=16,
    n_workers=4,
    device=None,
    key_added="anatomical_structures",
):
    if transform is None:
        transform = model.get_transform()

    postprocess_fn = model.get_postprocess_fn()

    dataset = wsi.ds.tile_images(tile_key=tile_key, transform=transform)
    dl = DataLoader(dataset, num_workers=n_workers, batch_size=batch_size)

    # Move model to device
    if device is not None:
        model.to(device)

    downsample = wsi.tile_spec(tile_key).base_downsample

    # Launch the process pool for postprocessing
    polygons = []
    names = []
    # Run the segmentation
    for chunk in dl:
        images = chunk["image"]
        xs, ys = np.asarray(chunk["x"]), np.asarray(chunk["y"])
        if device is not None:
            images = images.to(device)
        output = model.segment(images)

        polys, ns = batch_postprocess(output, xs, ys, postprocess_fn, downsample)
        polygons.extend(polys)
        names.extend(ns)

    # === Merge the polygons ===
    merger = PolygonMerger(polygons, names)
    merger.merge()
    # === Refresh the progress bar ===
    data = []
    for name, polys in merger.merged_polygons.items():
        for p in polys:
            data.append([p, name])

    cells_df = (
        gpd.GeoDataFrame(data, columns=["geometry", "names"])
        .explode()
        .reset_index(drop=True)
    )
    add_shapes(wsi, key_added, cells_df)

    return polygons, names


def batch_postprocess(output, xs, ys, postprocess_fn, downsample):
    polygons = []
    names = []
    for i, (mask, x, y) in enumerate(zip(output, xs, ys)):
        result = postprocess_fn(mask)
        polys = result["polygons"]
        ns = result["names"]

        for poly, n in zip(polys, ns):
            poly = translate(
                scale(poly, xfact=downsample, yfact=downsample, origin=(0, 0)), x, y
            )
            polygons.append(poly)
            names.append(n)
    return polygons, names
