from functools import cached_property

import cv2
import numpy as np
import pandas as pd
from wsidata import WSIData
from wsidata.io import update_shapes_data

from lazyslide._const import Key


def point2shape(
    wsi: WSIData,
    key: str = "tiles",
    groupby: str = None,
):
    pass


def tissue_props(
    wsi: WSIData,
    key: str = Key.tissue,
):
    """Compute a series of geometrical properties of tissue piecies

    - "area"
    - "area_filled"
    - "convex_area"
    - "solidity"
    - "convexity"
    - "axis_major_length"
    - "axis_minor_length"
    - "eccentricity"
    - "orientation"
    - "extent"
    - "perimeter"
    - "circularity"

    Parameters
    ----------
    wsi : :class:`WSIData <wsidata.WSIData>`
        The WSIData object.
    key : str
        The tissue key.

    Returns
    -------
    None

    .. note::
        The geometry features will be added to the :code:`tissues | {tissue_key}` table in the WSIData object.
        The columns will be named after the properties, e.g. `area`, `solidity`.

    Examples
    --------
    .. code-block:: python

        >>> import lazyslide as zs
        >>> wsi = zs.datasets.sample()
        >>> zs.pp.find_tissues(wsi)
        >>> zs.tl.tissue_props(wsi)
        >>> wsi['tissues']

    """

    props = []
    cnts = []
    for tissue_contour in wsi.iter.tissue_contours(key):
        cnt = tissue_contour.contour
        holes = tissue_contour.holes

        cnt_array = np.asarray(cnt.exterior.coords.xy, dtype=np.int32).T
        holes_array = [
            np.asarray(h.exterior.coords.xy, dtype=np.int32).T for h in holes
        ]

        _props = contour_props(cnt_array, holes_array)
        cnts.append(cnt)
        props.append(_props)

    props = pd.DataFrame(props).to_dict(orient="list")
    update_shapes_data(wsi, key, props)


class ContourProps:
    def __init__(self, cnt, holes=None):
        self.cnt = cnt
        self.holes = holes

    @cached_property
    def area_filled(self):
        return cv2.contourArea(self.cnt)

    @cached_property
    def area(self):
        """Area without holes."""
        if self.holes is None:
            return self.area_filled
        else:
            area = self.area_filled
            for hole in self.holes:
                area -= cv2.contourArea(hole)
            return area

    @cached_property
    def bbox(self):
        x, y, w, h = cv2.boundingRect(self.cnt)
        return x, y, w, h

    @cached_property
    def centroid(self):
        M = self.moments
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return cX, cY

    @cached_property
    def convex_hull(self):
        return cv2.convexHull(self.cnt)

    @cached_property
    def convex_area(self):
        return cv2.contourArea(self.convex_hull)

    @cached_property
    def solidity(self):
        """Solidity is the ratio of the contour area to the convex area."""
        if self.convex_area == 0:
            return 0
        return self.area / self.convex_area

    @cached_property
    def convexity(self):
        """Convexity is the ratio of the convex area to the contour area."""
        if self.area == 0:
            return 0
        return self.convex_area / self.area

    @cached_property
    def ellipse(self):
        return cv2.fitEllipse(self.cnt)

    @cached_property
    def axis_major_length(self):
        x1, x2 = self.ellipse[1]
        if x1 < x2:
            return x2
        return x1

    @cached_property
    def axis_minor_length(self):
        x1, x2 = self.ellipse[1]
        if x1 < x2:
            return x1
        return x2

    @cached_property
    def eccentricity(self):
        if self.axis_major_length == 0:
            return 0
        return np.sqrt(1 - (self.axis_minor_length**2) / (self.axis_major_length**2))

    @cached_property
    def orientation(self):
        return self.ellipse[2]

    @cached_property
    def extent(self):
        if self.area == 0:
            return 0
        return self.area / (self.bbox[2] * self.bbox[3])

    @cached_property
    def perimeter(self):
        return cv2.arcLength(self.cnt, True)

    @cached_property
    def circularity(self):
        if self.perimeter == 0:
            return 0
        return 4 * np.pi * self.area / (self.perimeter**2)

    @cached_property
    def moments(self):
        return cv2.moments(self.cnt)

    @cached_property
    def moments_hu(self):
        return cv2.HuMoments(self.moments)

    def __call__(self):
        props = {
            "area": self.area,
            "area_filled": self.area_filled,
            "convex_area": self.convex_area,
            "solidity": self.solidity,
            "convexity": self.convexity,
            "axis_major_length": self.axis_major_length,
            "axis_minor_length": self.axis_minor_length,
            "eccentricity": self.eccentricity,
            "orientation": self.orientation,
            "extent": self.extent,
            "perimeter": self.perimeter,
            "circularity": self.circularity,
        }

        for ix, box in enumerate(self.bbox):
            props[f"bbox-{ix}"] = box

        for ix, c in enumerate(self.centroid):
            props[f"centroid-{ix}"] = c

        for i, hu in enumerate(self.moments_hu):
            props[f"hu-{i}"] = hu[0]

        for key, value in self.moments.items():
            props[f"moment-{key}"] = value

        return props


def contour_props(cnt: np.ndarray, holes=None):
    """Calculate the properties of a contour."""
    return ContourProps(cnt, holes)()
