from __future__ import annotations

from typing import Sequence, Dict

import cv2
import numpy as np
from matplotlib import pyplot as plt
from shapely import Polygon


class Mask:
    def __init__(self, mask: np.ndarray):
        self.mask = mask

    @staticmethod
    def _detect_format(mask):
        mask_shape = mask.shape
        if len(mask_shape) == 2:
            n_classes = len(np.unique(mask))
            if n_classes == 2:
                format = "binary"
            elif n_classes > 2:
                format = "multilabel"
            else:
                raise ValueError("Invalid mask format.")
        else:
            format = "multiclass"
        return format

    @classmethod
    def from_polygons(cls, polygons):
        raise NotImplementedError()

    @classmethod
    def from_array(cls, mask):
        mask = np.asarray(mask)
        format = cls._detect_format(mask)
        if format == "binary":
            return BinaryMask(mask)
        elif format == "multilabel":
            return MultiLabelMask(mask)
        elif format == "multiclass":
            return MultiClassMask(mask)
        else:
            raise ValueError("Invalid mask format.")

    def to_polygons(
        self,
        min_area: float = 0,
        min_hole_area: float = 0,
        detect_holes: bool = True,
        ignore_index: int | Sequence[int] | None = None,
    ) -> Sequence[Polygon]:
        raise NotImplementedError()

    def to_binary_mask(self) -> np.ndarray:
        raise TypeError(f"Cannot convert {self.__class__.__name__} to binary mask.")

    def to_multilabel_mask(self) -> np.ndarray:
        raise TypeError(f"Cannot convert {self.__class__.__name__} to multilabel mask.")

    def to_multiclass_mask(self) -> np.ndarray:
        raise TypeError(f"Cannot convert {self.__class__.__name__} to multiclass mask.")

    def plot(self):
        _, ax = plt.subplots()
        ax.imshow(self.mask, cmap="Blues")
        return ax


class BinaryMask(Mask):
    def __init__(self, mask: np.ndarray):
        assert mask.ndim == 2, "Binary mask must be 2D."
        assert len(np.unique(mask)) == 2, "Binary mask must have 2 unique values."
        super().__init__(mask)

    def to_polygons(
        self,
        min_area: float = 0,
        min_hole_area: float = 0,
        detect_holes: bool = True,
        ignore_index: int | Sequence[int] | None = None,  # noqa
    ) -> Sequence[Polygon]:
        return _binary2polygons(
            self.mask,
            min_area=min_area,
            min_hole_area=min_hole_area,
            detect_holes=detect_holes,
        )

    def to_binary_mask(self) -> np.ndarray:
        return self.mask

    def to_multilabel_mask(self) -> np.ndarray:
        return self.mask

    def to_multiclass_mask(self) -> np.ndarray:
        return np.newaxis(self.mask, axis=0)


class MultiLabelMask(Mask):
    def __init__(self, mask: np.ndarray):
        assert mask.ndim == 2, "Multilabel mask must be 2D."
        super().__init__(mask)
        self.classes = np.sort(np.unique(self.mask))
        # drop the background class
        self.classes = self.classes[self.classes != 0]
        self.n_classes = len(self.classes)

    def to_polygons(
        self,
        min_area=0,
        min_hole_area=0,
        detect_holes: bool = True,
        ignore_index: int | Sequence[int] | None = None,
    ) -> Dict[int, Sequence[Polygon]]:
        if ignore_index is not None:
            if isinstance(ignore_index, int):
                ignore_index = {ignore_index}
            ignore_index = set(ignore_index)
        else:
            ignore_index = set()

        polys = {}
        for c in self.classes:
            if c in ignore_index:
                continue
            mask = np.asarray(self.mask == c, dtype=np.uint8)
            polys[c] = _binary2polygons(
                mask,
                min_area=min_area,
                min_hole_area=min_hole_area,
                detect_holes=detect_holes,
            )

        return polys

    def to_binary_mask(self) -> np.ndarray:
        mask = np.zeros(self.mask.shape, dtype=np.uint8)
        mask[self.mask != 0] = 1
        return mask

    def to_multiclass_mask(self) -> np.ndarray:
        mask = np.zeros((self.n_classes, *self.mask.shape), dtype=np.uint8)
        for i, c in enumerate(self.classes):
            mask[i] = self.mask == c
        return mask

    def to_multilabel_mask(self) -> np.ndarray:
        return self.mask


class MultiClassMask(Mask):
    def __init__(self, mask: np.ndarray, skip_bg=True):
        assert mask.ndim == 3, "Multiclass mask must be 3D."
        assert len(np.unique(mask)) <= 2, "Multiclass mask must only contains 0 and 1."
        super().__init__(mask.astype(np.uint8))

        if not skip_bg:
            self.n_classes = self.mask.shape[0]
            self.classes = np.arange(self.n_classes)
        else:
            self.n_classes = self.mask.shape[0] - 1
            self.classes = np.arange(1, self.n_classes + 1)

    def to_polygons(
        self,
        min_area=0,
        min_hole_area: int = 0,
        detect_holes: bool = True,
        ignore_index: int | Sequence[int] | None = None,
    ) -> Dict[int, Sequence[Polygon]]:
        if ignore_index is not None:
            if isinstance(ignore_index, int):
                ignore_index = {ignore_index}
            ignore_index = set(ignore_index)
        else:
            ignore_index = set()

        polys = {}
        for c in self.classes:
            if c in ignore_index:
                continue
            polys[c] = _binary2polygons(
                self.mask[c],
                min_area=min_area,
                min_hole_area=min_hole_area,
                detect_holes=detect_holes,
            )
        return polys

    def to_binary_mask(self) -> np.ndarray:
        mask = np.zeros(self.mask.shape[1:], dtype=np.uint8)
        mask[self.mask.sum(axis=0) > 0] = 1
        return mask

    def to_multilabel_mask(self) -> np.ndarray:
        mask = np.zeros(self.mask.shape[1:], dtype=np.uint8)
        for i, c in enumerate(self.classes):
            mask[self.mask[i] == 1] = c
        return mask

    def to_multiclass_mask(self) -> np.ndarray:
        return self.mask

    def plot(self):
        fig, axes = plt.subplots(1, self.n_classes)
        for i, ax in enumerate(axes):
            ax.imshow(self.mask[i], cmap="Blues")
        return fig


def _binary2polygons(
    mask,
    min_area: float = 0,
    min_hole_area: float = 0,
    detect_holes: bool = True,
) -> Sequence[Polygon]:
    """
    Convert binary mask to polygon.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask.
    min_area : int
        Minimum area of detected regions to be included in the polygon.
    min_hole_area : int
        Minimum area of detected holes to be included in the polygon.
    detect_holes : bool
        Whether to detect holes in regions.

    Returns
    -------
    List[Polygon]
        List of polygons.

    """
    if min_area < 1:
        min_area = int(min_area * mask.size)
    if min_hole_area < 1:
        min_hole_area = int(min_hole_area * mask.size)

    mode = cv2.RETR_CCOMP if detect_holes else cv2.RETR_EXTERNAL
    contours, hierarchy = cv2.findContours(
        mask, mode=mode, method=cv2.CHAIN_APPROX_NONE
    )

    if hierarchy is None:
        # no contours found --> return empty mask
        return []
    elif not detect_holes:
        # If we don't want to detect holes, we can simply return the contours
        polys = []
        cnt_id = 0
        for i, cnt in enumerate(contours):
            if cv2.contourArea(cnt) > min_area:
                cnt = np.squeeze(cnt, axis=1)
                # A polygon with less than 4 points is not valid
                if len(cnt) >= 4:
                    polys.append(Polygon(shell=cnt, holes=[]))
                    cnt_id += 1
        return polys
    else:
        # separate outside and inside contours (region boundaries vs. holes in regions)
        # find the outside contours by looking for those with no parents (4th column is -1 if no parent)
        # TODO: Handle nested contours
        poly_ixs = []
        for i, (cnt, hier) in enumerate(zip(contours, hierarchy[0])):
            # Check if the contour has a parent contour (i.e., if it's not a top-level contour)
            holes_ix = []
            if hier[3] == -1:
                area = cv2.contourArea(cnt)
                if area > min_area:
                    next_hole_index = hier[2]
                    # Iterate through the holes
                    while True:
                        # If it's a hole, add it to the list
                        if next_hole_index != -1:
                            next_hole = hierarchy[0][next_hole_index]
                            if (
                                cv2.contourArea(contours[next_hole_index])
                                > min_hole_area
                            ):
                                holes_ix.append(next_hole_index)
                            next_hole_index = next_hole[0]
                        else:
                            break
                    poly_ixs.append((i, holes_ix))

        polys = []
        for cnt_id, (cnt_ix, holes_ixs) in enumerate(poly_ixs):
            polys.append(
                Polygon(
                    shell=np.squeeze(contours[cnt_ix], axis=1),
                    holes=[np.squeeze(contours[ix], axis=1) for ix in holes_ixs],
                )
            )

        return polys
