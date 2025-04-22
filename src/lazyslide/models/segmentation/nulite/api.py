from typing import Literal

import cv2
import numpy as np
import torch
import geopandas as gpd
from torchvision.transforms.v2 import ToImage, ToDtype, Normalize, Compose
from huggingface_hub import hf_hub_download

from lazyslide.cv import Mask
from lazyslide.models.base import SegmentationModel

from .model import NuLite as NuLiteModel


class NuLite(SegmentationModel):
    def __init__(
        self,
        variant: Literal["H", "M", "T"] = "H",
    ):
        model_file = hf_hub_download(
            "RendeiroLab/LazySlide-models", f"nulite/NuLite-{variant}-Weights.pth"
        )

        weights = torch.load(model_file, map_location="cpu")

        config = weights["config"]
        self.model = NuLiteModel(
            config["data.num_nuclei_classes"],
            config["data.num_tissue_classes"],
            config["model.backbone"],
        )
        self.model.load_state_dict(weights["model_state_dict"])

    def get_transform(self):
        return Compose(
            [
                ToImage(),
                ToDtype(dtype=torch.float32, scale=True),
                Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

    @torch.inference_mode()
    def segment(self, image):
        return self.model.forward(image, retrieve_tokens=True)

    def get_postprocess(self):
        return nulite_preprocess


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
    nucleus_size: (int, int) = (20, 5000),
) -> gpd.GeoDataFrame:
    """Preprocess the image for NuLite model."""

    binary_mask = output["nuclei_binary_map"].softmax(0).detach().cpu().numpy()[1]
    hv_map = output["hv_map"].detach().cpu().numpy()
    type_prob_map = (
        output["nuclei_type_map"].softmax(0).detach().cpu().numpy()[1::]
    )  # to skip background

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

    unique_labels = np.unique(markers_clean)
    final_seg = np.zeros_like(markers_clean, dtype=np.int32)
    cells = []
    nucleus_size_min, nucleus_size_max = nucleus_size
    for lbl in unique_labels:
        if lbl <= 1:  # Skip background (-1) and unknown (1)
            continue
        mask = markers_clean == lbl
        x, y = np.where(mask)
        area = len(x)

        if nucleus_size_min <= area <= nucleus_size_max:
            probs = type_prob_map[:, x, y].mean(1)
            class_ix = np.argmax(probs)
            class_prob = type_prob_map[class_ix, x, y].mean()
            m = Mask.from_array(mask.astype(np.uint8))
            poly = m.to_polygons()[0]
            cells.append([CLASS_MAPPING[class_ix + 1], class_prob, poly])
            final_seg[markers_clean == lbl] = lbl
    return gpd.GeoDataFrame(cells, columns=["name", "prob", "geometry"])
