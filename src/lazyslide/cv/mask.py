from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Dict, Mapping, Sequence

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from legendkit import cat_legend
from matplotlib import pyplot as plt
from shapely import Polygon

from lazyslide._utils import find_stack_level

# Mask: (Any)
#   The base class for all mask types.
# BinaryMask: [C], H, W
#   A mask with foreground and background, can use with multiple classes.
# MultilabelMask: [C], H, W
#   A mask with multiple classes, where each pixel is assigned to multiple classes.
# MulticlassMask: H, W
#   A mask with multiple classes, where each pixel can only belong to one class.
# InstanceMap: H, W
#   A mask where each unique integer represents a different instance.
# ProbabilisticMap: [C], H, W
#   A mask that includes probability information for each pixel, can be used with multiple classes.


class Mask(ABC):
    def __init__(
        self,
        mask: np.ndarray,
        prob_map: np.ndarray | None = None,
        class_names: Sequence[str] | Mapping[int, str] = None,
    ):
        self.mask = mask
        self.prob_map = prob_map
        # Parse class names if provided
        if class_names is not None:
            if isinstance(class_names, Mapping):
                self.class_names = class_names
            elif isinstance(class_names, Sequence):
                self.class_names = {i: name for i, name in enumerate(class_names)}
            else:
                raise ValueError("class_name must be a Mapping or a Sequence.")
        else:
            self.class_names = None

    @staticmethod
    def _is_integer_dtype(mask):
        return np.issubdtype(mask.dtype, np.integer)

    @staticmethod
    def _is_probability_map(prob_map):
        # Check dtype
        is_float = np.issubdtype(prob_map.dtype, np.floating)
        if not is_float:
            return False
        # Check values
        bt_1 = np.min(prob_map) >= 0
        lt_1 = np.max(prob_map) <= 1
        if not (bt_1 and lt_1):
            return False
        return True

    @classmethod
    def from_polygons(
        cls,
        polygons: gpd.GeoDataFrame,
        bounding_box=None,
        class_col=None,
        classes_order=None,
    ):
        """
        Create a mask from polygons.

        Parameters
        ----------
        polygons : gpd.GeoDataFrame
            GeoDataFrame containing polygons.
        bounding_box : tuple, optional
            Bounding box to define the mask size (xmin, ymin, xmax, ymax).
        class_col : str, optional
            The column name in polygons that contain class labels.
        classes_order : list, optional
            Order of classes in the mask.

        Returns
        -------
        Mask
            An instance of the appropriate Mask subclass.
        """
        if bounding_box is None:
            bounding_box = polygons.total_bounds
        if class_col is not None:
            classes = polygons[class_col].unique()
            if classes_order is None:
                classes_order = sorted(classes)
            classes_ix = {c: i for i, c in enumerate(classes_order)}
            n_classes = len(classes)
        else:
            n_classes = 1
        # Create an empty mask
        mask_shape = (
            n_classes,
            int(bounding_box[3] - bounding_box[1]),
            int(bounding_box[2] - bounding_box[0]),
        )
        mask = np.zeros(mask_shape, dtype=np.uint8)

        for _, row in polygons.iterrows():
            poly = row["geometry"]
            if class_col is not None:
                class_name = row[class_col]
                class_index = classes_ix[class_name]
                # Create a mask for the polygon
                poly_mask = np.zeros(mask_shape[1:], dtype=np.uint8)
                cv2.fillPoly(
                    poly_mask, [np.array(poly.exterior.coords).astype(np.int32)], 1
                )
            if poly.interiors:
                for hole in poly.interiors:
                    cv2.fillPoly(poly_mask, [np.array(hole.coords).astype(np.int32)], 0)
            mask[class_index] += poly_mask
        if n_classes == 1:
            mask = mask.squeeze(0)  # Remove the class dimension if only one class
            return BinaryMask(mask)
        else:
            # Ensure the mask is of an integer type
            mask = np.asarray(mask, dtype=np.uint8)
            return MultilabelMask(mask, class_names=classes_order)

    @abstractmethod
    def to_polygons(
        self,
        min_area: float = 0,
        min_hole_area: float = 0,
        detect_holes: bool = True,
        ignore_index: int | Sequence[int] | None = None,
    ) -> gpd.GeoDataFrame:
        raise NotImplementedError()

    def to_binary_mask(self) -> np.ndarray:
        raise NotImplementedError(
            f"Cannot convert {self.__class__.__name__} to binary mask."
        )

    def to_multilabel_mask(self) -> np.ndarray:
        raise NotImplementedError(
            f"Cannot convert {self.__class__.__name__} to multilabel mask."
        )

    def to_multiclass_mask(self) -> np.ndarray:
        raise NotImplementedError(
            f"Cannot convert {self.__class__.__name__} to multiclass mask."
        )

    def to_instance_map(self) -> np.ndarray:
        raise NotImplementedError(
            f"Cannot convert {self.__class__.__name__} to instance mask."
        )

    def to(self, T: str | Mask) -> np.ndarray:
        """
        Convert the mask to the specified type.

        Parameters
        ----------
        T : {"binary", "multilabel", "multiclass", "instance"} or Mask type
            The mask type to convert to.

        Returns
        -------
        Mask
            An instance of the specified format.
        """
        if isinstance(T, Mask):
            T = T.__class__.__name__.lower().strip("mask")
        if isinstance(T, str):
            if T == "binary":
                return self.to_binary_mask()
            elif T == "multilabel":
                return self.to_multilabel_mask()
            elif T == "multiclass":
                return self.to_multiclass_mask()
            elif T == "instance":
                return self.to_instance_map()
            else:
                raise ValueError(f"Unknown mask type: {T}")
        else:
            raise TypeError("Mask type must be a string or a Mask subclass.")

    def plot(self, ax=None, **kwargs):
        ax = ax or plt.gca()
        ax.imshow(self.mask, **kwargs)
        return ax


class BinaryMask(Mask):
    def __init__(
        self,
        mask: np.ndarray,
        prob_map: np.ndarray | None = None,
        class_names: Sequence[str] | Mapping[int, str] = None,
    ):
        assert mask.ndim == 2, "Binary mask must be 2D."
        if prob_map is not None:
            assert prob_map.shape == mask.shape, (
                "Probability mask must have the same shape as the binary mask."
            )
        # Coerce the mask to binary (0 and 1)
        mask = np.asarray(mask > 0, dtype=np.uint8)
        super().__init__(mask, prob_map, class_names)

    def to_polygons(
        self,
        min_area: float = 0,
        min_hole_area: float = 0,
        detect_holes: bool = True,
        ignore_index: int | Sequence[int] | None = None,  # noqa
    ) -> gpd.GeoDataFrame:
        return binary_mask_to_polygons_with_prob(
            self.mask,
            prob_map=self.prob_map,
            min_area=min_area,
            min_hole_area=min_hole_area,
            detect_holes=detect_holes,
        )

    def to_binary_mask(self) -> np.ndarray:
        return self.mask

    def to_multilabel_mask(self) -> np.ndarray:
        return np.expand_dims(self.mask, axis=0)

    def to_multiclass_mask(self) -> np.ndarray:
        return self.mask

    def to_instance_map(self) -> np.ndarray:
        _, instance_map = cv2.connectedComponents(
            self.mask, connectivity=8, ltype=cv2.CV_32S
        )
        return instance_map


class MulticlassMask(Mask):
    def __init__(
        self,
        mask: np.ndarray,
        prob_map: np.ndarray | None = None,
        class_names: Sequence[str] | Mapping[int, str] = None,
    ):
        assert mask.ndim == 2, "Multiclass mask must be 2D."
        assert self._is_integer_dtype(mask), "Multiclass mask must be of integer type."
        if prob_map is not None:
            assert prob_map.shape == mask.shape, (
                "Probability mask must have the same shape as the multiclass mask."
            )
        super().__init__(mask, prob_map)
        self.classes = np.sort(np.unique(self.mask))
        self.n_classes = len(self.classes)
        if class_names is not None:
            if isinstance(class_names, Mapping):
                self.class_names = class_names
            elif isinstance(class_names, Sequence):
                self.class_names = {i: name for i, name in enumerate(class_names)}
            else:
                raise ValueError("class_name must be a Mapping or a Sequence.")

    def to_polygons(
        self,
        min_area=0,
        min_hole_area=0,
        detect_holes: bool = True,
        ignore_index: int | Sequence[int] | None = 0,
    ) -> GeoDataFrame | None:
        if ignore_index is not None:
            if isinstance(ignore_index, int):
                ignore_index = {ignore_index}
            ignore_index = set(ignore_index)
        else:
            ignore_index = set()

        polys = []
        for c in self.classes:
            if c in ignore_index:
                continue
            mask = np.asarray(self.mask == c, dtype=np.uint8)
            polys_c = binary_mask_to_polygons_with_prob(
                mask,
                prob_map=self.prob_map,
                min_area=min_area,
                min_hole_area=min_hole_area,
                detect_holes=detect_holes,
            )
            if len(polys_c) > 0:
                polys_c["class"] = c
                polys.append(polys_c)
        if len(polys) == 0:
            return gpd.GeoDataFrame()
        final = gpd.GeoDataFrame(pd.concat(polys, ignore_index=True)).reset_index(
            drop=True
        )
        if self.class_names is not None:
            final["class"] = final["class"].map(self.class_names)
        return final

    def to_binary_mask(self) -> np.ndarray:
        mask = np.zeros(self.mask.shape, dtype=np.uint8)
        mask[self.mask != 0] = 1
        return mask

    def to_multiclass_mask(self) -> np.ndarray:
        return self.mask

    def to_multilabel_mask(self) -> np.ndarray:
        mask = np.zeros((self.n_classes, *self.mask.shape), dtype=np.uint8)
        for i, c in enumerate(self.classes):
            mask[i] = self.mask == c
        return mask

    def plot(self, ax=None, **kwargs):
        ax = ax or plt.gca()
        sm = ax.imshow(self.mask, **kwargs)
        # Add legend if class names are provided
        if self.class_name is not None:
            labels = [self.class_name.get(c, str(c)) for c in self.classes]
            colors = [sm.cmap(sm.norm(c)) for c in self.classes]
            cat_legend(labels=labels, colors=colors, ax=ax, loc="out right center")
        ax.set_title("Multiclass Mask")
        return ax


class MultilabelMask(Mask):
    def __init__(
        self,
        mask: np.ndarray,
        prob_map: np.ndarray | None = None,
        class_names: Sequence[str] | Mapping[int, str] = None,
    ):
        assert mask.ndim == 3, "Multiclass mask must be C, H, W."
        assert self._is_integer_dtype(mask), "Multiclass mask must be of integer type."
        mask = np.asarray(mask > 0, dtype=np.uint8)
        super().__init__(mask, prob_map, class_names)
        self.n_classes = self.mask.shape[0]
        self.classes = np.arange(self.n_classes)

    def to_polygons(
        self,
        min_area=0,
        min_hole_area: int = 0,
        detect_holes: bool = True,
        ignore_index: int | Sequence[int] | None = 0,
    ) -> gpd.GeoDataFrame:
        if ignore_index is not None:
            if isinstance(ignore_index, int):
                ignore_index = {ignore_index}
            ignore_index = set(ignore_index)
        else:
            ignore_index = set()

        polys = []
        for c in self.classes:
            if c in ignore_index:
                continue
            polys_c = binary_mask_to_polygons_with_prob(
                self.mask[c],
                prob_map=self.prob_map[c] if self.prob_map is not None else None,
                min_area=min_area,
                min_hole_area=min_hole_area,
                detect_holes=detect_holes,
            )
            if len(polys_c) > 0:
                polys_c["class"] = c
                polys.append(polys_c)
        if len(polys) == 0:
            return gpd.GeoDataFrame()
        final = gpd.GeoDataFrame(pd.concat(polys, ignore_index=True)).reset_index(
            drop=True
        )
        if self.class_names is not None:
            final["class"] = final["class"].map(self.class_names)
        return final

    def to_binary_mask(self) -> np.ndarray:
        mask = np.zeros(self.mask.shape[1:], dtype=np.uint8)
        mask[self.mask.sum(axis=0) > 0] = 1
        return mask

    def to_multilabel_mask(self) -> np.ndarray:
        return self.mask

    def to_multiclass_mask(self) -> np.ndarray:
        mask = np.zeros(self.mask.shape[1:], dtype=np.uint8)
        for i, c in enumerate(self.classes):
            mask[self.mask[i] == 1] = c
        return mask

    def plot(self, fig=None, **kwargs):
        fig = fig or plt.gcf()
        axes = fig.subplots(nrows=1, ncols=self.n_classes)
        fig.set_size_inches(self.n_classes * 3, 3)
        for i, ax in enumerate(axes):
            ax.imshow(self.mask[i], **kwargs)
            if self.class_names is not None:
                title = self.class_names.get(i, str(i))
            else:
                title = f"Class {i}"
            ax.set_title(title)
        return fig


class InstanceMap(Mask):
    """The class for instance mask."""

    def __init__(
        self,
        instance_map: np.ndarray,
        prob_map: np.ndarray | None = None,
        class_names: Sequence[str] | Mapping[int, str] = None,
    ):
        assert instance_map.ndim == 2, "Instance map must be 2D."
        # The map must be an integer type with unique values for each instance
        assert np.issubdtype(instance_map.dtype, np.integer), (
            "Instance map must be of integer type."
        )

        self._is_classification = False
        if prob_map is not None:
            # If the probability map is provided, it must be 2D or 3D
            # 2D is for cell probabilities
            # 3D is for cell classification
            self._is_classification = prob_map.ndim == 3

        super().__init__(instance_map, prob_map, class_names)

    def to_polygons(
        self,
        min_area: float = 0,
        min_hole_area: float = 0,
        detect_holes: bool = True,
        ignore_index: int | Sequence[int] | None = None,
    ) -> gpd.GeoDataFrame:
        """
        Convert instance mask to polygons.

        Parameters
        ----------
        min_area : float
            Minimum area of detected regions to be included in the polygon.
        min_hole_area : float
            Minimum area of detected holes to be included in the polygon.
        detect_holes : bool
            Whether to detect holes in regions.
        ignore_index : int or Sequence[int] or None
            Indexes to ignore.

        Returns
        -------
        Dict[int, Sequence[Polygon]]
            Dictionary of polygons for each instance.

        """
        if ignore_index is not None:
            raise ValueError("ignore_index is not supported for InstanceMap.")
        instance_ids = np.unique(self.mask)
        instances = []
        for instance_id in instance_ids:
            if instance_id <= 0:  # Skip background
                continue
            instance_mask = np.asarray(self.mask == instance_id, dtype=np.uint8)
            polys = binary_mask_to_polygons_with_prob(
                instance_mask,
                prob_map=self.prob_map,
                min_area=min_area,
                min_hole_area=min_hole_area,
                detect_holes=detect_holes,
            )
            if len(polys) > 0:
                instances.append(polys.iloc[0])
        if len(instances) == 0:
            return gpd.GeoDataFrame()
        # Create a GeoDataFrame from the instances
        instances = gpd.GeoDataFrame(instances)
        if "class" in instances.columns:
            if self.class_names is not None:
                instances["class"] = instances["class"].map(self.class_names)
        return instances


class ProbabilityMap(Mask):
    def __init__(
        self,
        probability_map: np.ndarray,
        prob_map: np.ndarray | None = None,
        class_names: Sequence[str] | Mapping[int, str] = None,
    ):
        # The probability map can be 2D or 3D, but must be of the floating point type
        assert probability_map.ndim in (2, 3), "Probability map must be 2D or 3D."
        assert self._is_probability_map(probability_map), (
            "Probability map must be of floating "
            "point type with values between 0 and 1."
        )
        if prob_map is not None:
            warnings.warn(
                "prob_map is not used in ProbabilityMap, it will be ignored.",
                stacklevel=find_stack_level(),
            )
            prob_map = None
        super().__init__(probability_map, prob_map, class_names)
        self.is2D = probability_map.ndim == 2

    def to_polygons(
        self,
        threshold: float = 0.5,
        min_area: float = 0,
        min_hole_area: float = 0,
        detect_holes: bool = True,
        ignore_index: int | Sequence[int] | None = 0,
    ) -> gpd.GeoDataFrame:
        """
        Convert the probability map to polygons.

        Parameters
        ----------
        threshold : float
            Threshold to convert the probability map to binary mask.
        min_area : float
            Minimum area of detected regions to be included in the polygon.
        min_hole_area : float
            Minimum area of detected holes to be included in the polygon.
        detect_holes : bool
            Whether to detect holes in regions.
        ignore_index : int or Sequence[int] or None
            Indexes to ignore.

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame containing polygons and their probabilities.

        """
        if self.is2D:
            binary_mask = (self.mask > threshold).astype(np.uint8)
            return binary_mask_to_polygons_with_prob(
                binary_mask,
                prob_map=self.mask,
                min_area=min_area,
                min_hole_area=min_hole_area,
                detect_holes=detect_holes,
            )
        else:
            if ignore_index is not None:
                if isinstance(ignore_index, int):
                    ignore_index = {ignore_index}
                ignore_index = set(ignore_index)
            else:
                ignore_index = set()
            # For 3D probability maps, we need to process each channel separately
            polys = []
            for i in range(self.mask.shape[0]):
                if i in ignore_index:
                    continue
                binary_mask = (self.mask[i] > threshold).astype(np.uint8)
                polys_c = binary_mask_to_polygons_with_prob(
                    binary_mask,
                    prob_map=self.mask[i],
                    min_area=min_area,
                    min_hole_area=min_hole_area,
                    detect_holes=detect_holes,
                )
                if len(polys_c) > 0:
                    polys_c["class"] = i
                    polys.append(polys_c)
            if len(polys) == 0:
                return gpd.GeoDataFrame()
            final = gpd.GeoDataFrame(pd.concat(polys, ignore_index=True)).reset_index(
                drop=True
            )
            if self.class_names is not None:
                final["class"] = final["class"].map(self.class_names)
            return final


def binary_mask_to_polygons(
    binary_mask,
    min_area: float = 0,
    min_hole_area: float = 0,
    detect_holes: bool = True,
) -> Sequence[Polygon]:
    """
    Convert binary mask to polygon.

    Parameters
    ----------
    binary_mask : np.ndarray
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
        min_area = int(min_area * binary_mask.size)
    if min_hole_area < 1:
        min_hole_area = int(min_hole_area * binary_mask.size)

    mode = cv2.RETR_CCOMP if detect_holes else cv2.RETR_EXTERNAL
    contours, hierarchy = cv2.findContours(
        binary_mask, mode=mode, method=cv2.CHAIN_APPROX_NONE
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


def binary_mask_to_polygons_with_prob(
    binary_mask,
    prob_map=None,
    min_area: float = 0,
    min_hole_area: float = 0,
    detect_holes: bool = True,
) -> gpd.GeoDataFrame:
    """
    Convert binary mask to polygon and include probability information.

    Parameters
    ----------
    binary_mask : np.ndarray
        Binary mask.
    prob_map : np.ndarray, optional
        Probability mask with the same shape as the binary mask.
    min_area : float
        Minimum area of detected regions to be included in the polygon.
    min_hole_area : float
        Minimum area of detected holes to be included in the polygon.
    detect_holes : bool
        Whether to detect holes in regions.

    Returns
    -------
    :class:`GeoDataFrame <geopandas.GeoDataFrame>`
        GeoDataFrame containing polygons and their probabilities.

    """
    # Get polygons using the existing function
    polys = binary_mask_to_polygons(
        binary_mask,
        min_area=min_area,
        min_hole_area=min_hole_area,
        detect_holes=detect_holes,
    )

    # If no polygons were found, return an empty GeoDataFrame
    if not polys:
        return gpd.GeoDataFrame(columns=["geometry", "prob"])

    # Create a list to store data for the GeoDataFrame
    data = []

    # If a probability mask is provided, calculate the probability for each polygon
    if prob_map is not None:
        is_classification = prob_map.ndim == 3
        for poly in polys:
            # Create a mask for the current polygon
            poly_mask = np.zeros_like(binary_mask, dtype=np.uint8)
            # Convert polygon coordinates to integer points for cv2.fillPoly
            points = np.array(poly.exterior.coords, dtype=np.int32)
            # cv2.fillPoly(poly_mask, [points], 1)
            cv2.drawContours(poly_mask, [points], -1, 1, thickness=cv2.FILLED)
            # Fill the holes with 0 if detect_holes is True
            if detect_holes:
                for hole in poly.interiors:
                    hole_points = np.array(hole.coords, dtype=np.int32)
                    cv2.fillPoly(poly_mask, [hole_points], 0)

            # Calculate mean probability within the polygon
            if is_classification:
                # If it's a classification map, calculate the mean probability for each class
                # And then argmax to get the most probable class
                masked_prob = prob_map * poly_mask
                prob = (
                    np.sum(masked_prob, axis=(1, 2)) / np.sum(poly_mask)
                    if np.sum(poly_mask) > 0
                    else 0
                )
                c = np.argmax(prob)
                prob = prob[c]
                data.append({"geometry": poly, "prob": prob, "class": c})
            else:
                masked_prob = prob_map * poly_mask
                prob = (
                    np.sum(masked_prob) / np.sum(poly_mask)
                    if np.sum(poly_mask) > 0
                    else 0
                )
                # Add polygon and probability to data
                data.append({"geometry": poly, "prob": prob})
    else:
        for poly in polys:
            data.append({"geometry": poly})

    # Create and return GeoDataFrame
    return gpd.GeoDataFrame(data)
