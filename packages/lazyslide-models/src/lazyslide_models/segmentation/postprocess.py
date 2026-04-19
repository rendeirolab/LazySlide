import cv2
import geopandas as gpd
import numpy as np


def instanseg_postprocess(
    mask: np.ndarray,
):
    """
    Postprocess the mask to get the cell polygons.

    The feature of each cell is average-pooling the feature map within the cell's bounding box.

    Parameters
    ----------
    mask: np.ndarray
        The mask array.

    """
    from lazyslide.cv import MultiLabelMask

    mmask = MultiLabelMask(mask)
    polys = mmask.to_polygons(min_area=5, detect_holes=False)
    cells = []
    for k, vs in polys.items():
        if len(vs) == 0:
            continue
        elif len(vs) == 1:
            cell = vs[0]
        else:
            # Get the largest polygon
            svs = sorted(vs, key=lambda x: x.area)
            cell = svs[-1]

        cells.append(cell)

    container = {"geometry": cells}
    return gpd.GeoDataFrame(container)


def semanticseg_postprocess(
    probs: np.ndarray,
    ignore_index: list[int] = None,
    min_area: int = 5,
    mapping: dict = None,
):
    from lazyslide.cv import MultiLabelMask

    mask = np.argmax(probs, axis=0).astype(np.uint8)
    mmask = MultiLabelMask(mask)
    polys = mmask.to_polygons(ignore_index=ignore_index, min_area=min_area)
    data = []
    for k, vs in polys.items():
        for v in vs:
            empty_mask = np.zeros_like(mask)

            cv2.drawContours(  # noqa
                empty_mask,
                [np.array(v.exterior.coords).astype(np.int32)],
                -1,
                1,
                thickness=cv2.FILLED,
            )

            prob = np.mean(probs[k][empty_mask == 1])
            class_name = k
            if mapping is not None:
                class_name = mapping[k]
            data.append([class_name, prob, v])

    return gpd.GeoDataFrame(data, columns=["class", "prob", "geometry"])
