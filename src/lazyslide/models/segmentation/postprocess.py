import numpy as np


def cellseg_postprocess(
    mask: np.ndarray,
    class_map: np.ndarray = None,
    feature_map: np.ndarray = None,
):
    """
    Postprocess the mask to get the cell polygons.

    The feature of each cell is average-pooling the feature map within the cell's bounding box.

    Parameters
    ----------
    mask: np.ndarray
        The mask array.
    class_map: np.ndarray, optional
        The class map array, by default None.
    feature_map: np.ndarray, optional
        The feature map array, by default None.

    """
    from lazyslide.cv import MultiLabelMask

    mask = MultiLabelMask(mask)
    polys = mask.to_polygons(min_area=5, detect_holes=False)
    cells = []
    # Optional
    names = []
    features = []
    for k, vs in polys.items():
        if len(vs) == 0:
            continue
        elif len(vs) == 1:
            cell = vs[0]
        else:
            # Get the largest polygon
            svs = sorted(vs, key=lambda x: x.area)
            cell = svs[-1]

        # Get the name of the cell
        mid_point = cell.centroid
        x, y = int(mid_point.x), int(mid_point.y)
        if class_map is not None:
            name = class_map[x, y]
            names.append(name)
        if feature_map is not None:
            xmin, ymin, xmax, ymax = cell.bounds
            feature = feature_map[xmin:xmax, ymin:ymax].mean(0)
            features.append(feature)

        cells.append(cell)

    container = {"polygons": cells}
    if len(names) > 0:
        container["names"] = names
    if len(features) > 0:
        container["features"] = np.vstack(features)

    return container


def semanticseg_postprocess(
    mask: np.ndarray,
    skip_bg: bool = True,
):
    from lazyslide.cv import MultiClassMask

    mask = MultiClassMask(mask)
    polys = mask.to_polygons()
    domains = []
    names = []
    for k, vs in polys.items():
        for v in vs:
            domains.append(v)
            names.append(k)

    return {"polygons": domains, "names": names}
