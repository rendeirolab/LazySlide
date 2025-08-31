from typing import Literal

import cv2
import geopandas as gpd
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import binary_fill_holes, measurements
from skimage.segmentation import watershed

from ..base import ModelTask, SegmentationModel


class NuLite(SegmentationModel, key="nulite"):
    task = ModelTask.segmentation
    license = ["Apache 2.0", "CC-BY-NC-SA-4.0"]
    description = "Nuclei instance segmentation and classification"
    commercial = False
    github_url = "https://github.com/CosmoIknosLab/NuLite"
    paper_url = "https://doi.org/10.48550/arXiv.2408.01797"
    bib_key = "Tommasino2024-tg"
    param_size = "47.9M"

    def __init__(
        self,
        variant: Literal["H", "M", "T"] = "H",
    ):
        from huggingface_hub import hf_hub_download

        model_file = hf_hub_download(
            "RendeiroLab/LazySlide-models", f"nulite/NuLite_{variant}_jit.pt"
        )

        self.model = torch.jit.load(model_file, map_location="cpu")

    def get_transform(self):
        from torchvision.transforms.v2 import Compose, Normalize, ToDtype, ToImage

        return Compose(
            [
                ToImage(),
                ToDtype(dtype=torch.float32, scale=True),
                Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

    # @torch.inference_mode()
    def segment(self, image):
        with torch.inference_mode():
            output = self.model(image)
        # return output
        # postprocess the output
        flattened = [
            dict(zip(output.keys(), values)) for values in zip(*output.values())
        ]

        instances_maps = []
        prob_maps = []
        for batch in flattened:
            instance_map = nulite_preprocess(batch)  # Numpy array
            prob_map = (
                batch["nuclei_type_map"].softmax(0).detach().cpu().numpy()
            )  # Skip background
            instances_maps.append(instance_map)
            prob_maps.append(prob_map)

        return {
            "instance_map": np.array(instances_maps),
            "class_map": np.array(prob_maps),
        }

    def supported_output(self):
        return ["instance_map", "class_map"]


CLASS_MAPPING = {
    0: "Background",
    1: "Neoplastic",
    2: "Inflammatory",
    3: "Connective",
    4: "Dead",
    5: "Epithelial",
}


def nulite_preprocess(
    output,
    ksize: int = 11,
    min_object_size: int = 3,
    nucleus_size: (int, int) = (20, 1000),
) -> gpd.GeoDataFrame:
    """Preprocess the image for NuLite model."""

    binary_mask = output["nuclei_binary_map"].softmax(0).detach().cpu().numpy()[1]
    hv_map = output["hv_map"].detach().cpu().numpy()
    # type_prob_map = (
    #     output["nuclei_type_map"].softmax(0).detach().cpu().numpy()[1::]
    # )  # to skip background

    _, blb = cv2.threshold(binary_mask.astype(np.float32), 0.5, 1, cv2.THRESH_BINARY)
    blb = blb.astype(np.uint8)

    # Remove small objects based on connected components.
    # Use cv2.connectedComponentsWithStats to label regions and filter by area.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(blb, connectivity=8)
    min_size = 3  # Minimum pixel area to keep an object
    blb_clean = np.zeros_like(blb)
    for label in range(1, num_labels):  # label 0 is the background.
        if stats[label, cv2.CC_STAT_AREA] >= min_size:
            blb_clean[labels == label] = 1

    h_map, v_map = hv_map

    # magnitude = np.sqrt(hv_map[0] ** 2 + hv_map[1] ** 2)
    # mask = magnitude > 0.8 # Threshold to filter out weak responses
    # # Apply the mask to the horizontal and vertical maps
    # h_map = np.where(mask, h_map, 0)
    # v_map = np.where(mask, v_map, 0)

    # STEP 2: Normalize directional maps
    h_dir_norm = cv2.normalize(
        h_map, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX
    ).astype(np.float32)
    v_dir_norm = cv2.normalize(
        v_map, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX
    ).astype(np.float32)

    # STEP 3: Compute edges using Sobel operators
    # ksize = 11  # Kernel size for Sobel operators; adjust for edge sensitivity.
    sobelh = cv2.Sobel(h_dir_norm, cv2.CV_64F, dx=1, dy=0, ksize=ksize)
    sobelv = cv2.Sobel(v_dir_norm, cv2.CV_64F, dx=0, dy=1, ksize=ksize)

    # Normalize the edge responses and invert them to prepare for the "distance" map.
    sobelh_norm = 1 - cv2.normalize(
        sobelh, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX
    )
    sobelv_norm = 1 - cv2.normalize(
        sobelv, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX
    )

    # Combine edge images by taking the maximum value at each pixel.
    overall = np.maximum(sobelh_norm, sobelv_norm)

    # Remove non-nuclei regions from the edge map.
    overall = overall - (1 - blb_clean.astype(np.float32))
    overall[overall < 0] = 0  # Set negative values to zero

    # STEP 4: Create an inverse “distance” map for watershed
    # The idea is to make the centers of nuclei correspond to local minima.
    # dist = (1.0 - overall) * blb_clean.astype(np.float32)
    # dist = -cv2.GaussianBlur(dist, (3, 3), 0)

    # STEP 5: Create markers for watershed (seed regions)
    # Identify the nucleus interior by thresholding the overall edge image.
    _, overall_bin = cv2.threshold(overall, 0.4, 1, cv2.THRESH_BINARY)
    overall_bin = overall_bin.astype(np.uint8)

    # Subtract the boundaries from the clean binary mask
    marker = blb_clean - overall_bin
    marker[marker < 0] = 0

    # Fill holes and do a morphological closing to smooth marker regions.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    marker_closed = cv2.morphologyEx(marker, cv2.MORPH_CLOSE, kernel)

    # Again, remove tiny markers using connected component analysis.
    num_labels, markers, stats, _ = cv2.connectedComponentsWithStats(
        marker_closed, connectivity=8
    )
    object_size = 10  # Minimum size (in pixels) for a marker region
    markers_clean = np.zeros_like(markers, dtype=np.int32)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= object_size:
            markers_clean[markers == label] = label

    # STEP 6: Apply the Watershed algorithm using only OpenCV
    # The watershed function in OpenCV requires a 3-channel image.
    # Here, we build a dummy 3-channel (RGB) image from our binary mask (for visualization/masking purposes).
    dummy_img = cv2.cvtColor((blb_clean * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # Watershed modifies the marker image in place.
    # The boundaries between segmented regions will be marked with -1.
    cv2.watershed(dummy_img, markers_clean)
    return markers_clean

    # unique_labels = np.unique(markers_clean)
    # final_seg = np.zeros_like(markers_clean, dtype=np.int32)
    # cells = []
    # nucleus_size_min, nucleus_size_max = nucleus_size
    # for lbl in unique_labels:
    #     if lbl <= 1:  # Skip background (-1) and unknown (1)
    #         continue
    #     mask = markers_clean == lbl
    #     x, y = np.where(mask)
    #     area = len(x)
    #
    #     if nucleus_size_min <= area <= nucleus_size_max:
    #         probs = type_prob_map[:, x, y].mean(1)
    #         class_ix = np.argmax(probs)
    #         class_prob = type_prob_map[class_ix, x, y].mean()
    #         m = Mask.from_array(mask.astype(np.uint8))
    #         poly = m.to_polygons()[0]
    #         cells.append([CLASS_MAPPING[class_ix + 1], class_prob, poly])
    #         final_seg[markers_clean == lbl] = lbl
    # return final_seg
    # return gpd.GeoDataFrame(cells, columns=["name", "prob", "geometry"])


def remove_small_objects(pred, min_size=64, connectivity=1):
    """Remove connected components smaller than the specified size.

    This function is taken from skimage.morphology.remove_small_objects, but the warning
    is removed when a single label is provided.

    Args:
        pred: input labelled array
        min_size: minimum size of instance in output array
        connectivity: The connectivity defining the neighborhood of a pixel.

    Returns:
        out: output array with instances removed under min_size

    """
    out = pred

    if min_size == 0:  # shortcut for efficiency
        return out

    if out.dtype == bool:
        selem = ndimage.generate_binary_structure(pred.ndim, connectivity)
        ccs = np.zeros_like(pred, dtype=np.int32)
        ndimage.label(pred, selem, output=ccs)
    else:
        ccs = out

    try:
        component_sizes = np.bincount(ccs.ravel())
    except ValueError:
        raise ValueError(
            "Negative value labels are not supported. Try "
            "relabeling the input with `scipy.ndimage.label` or "
            "`skimage.morphology.label`."
        )

    too_small = component_sizes < min_size
    too_small_mask = too_small[ccs]
    out[too_small_mask] = 0

    return out


def np_hv_postprocess(
    output: dict,
    ksize: int = 11,
    object_size: int = 3,
):
    binary_mask = output["nuclei_binary_map"].softmax(0).detach().cpu().numpy()[1]
    hv_map = output["hv_map"].detach().cpu().numpy()
    # type_prob_map = (
    #     output["nuclei_type_map"].softmax(0).detach().cpu().numpy()[1::]
    # )  # to skip background

    blb = np.array(binary_mask > 0.5, dtype=np.int32)
    blb = measurements.label(blb)[0]
    blb = remove_small_objects(blb)
    blb[blb > 0] = 1

    h_map, v_map = hv_map
    h_dir = cv2.normalize(
        h_map, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )
    v_dir = cv2.normalize(
        v_map, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )

    sobelh = cv2.Sobel(h_dir, cv2.CV_64F, dx=1, dy=0, ksize=ksize)
    sobelv = cv2.Sobel(v_dir, cv2.CV_64F, dx=0, dy=1, ksize=ksize)

    overall = np.maximum(sobelh, sobelv)
    overall = overall - (1 - blb)
    overall[overall < 0] = 0

    dist = (1.0 - overall) * blb
    dist = -cv2.GaussianBlur(dist, (3, 3), 0)

    overall = np.array(overall >= 0.4, dtype=np.int32)

    marker = blb - overall
    marker[marker < 0] = 0
    marker = binary_fill_holes(marker).astype("uint8")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
    marker = measurements.label(marker)[0]
    marker = remove_small_objects(marker, min_size=object_size)

    proced_pred = watershed(dist, markers=marker, mask=blb)

    return proced_pred
