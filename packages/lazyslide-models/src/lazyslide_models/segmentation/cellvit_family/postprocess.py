import cv2
import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_fill_holes, label
from skimage.segmentation import watershed

# def cellvit_postprocess(
#     nuclei_binary_map: np.ndarray,
#     hv_map: np.ndarray,
#     ksize: int = 11,
# ) -> gpd.GeoDataFrame:
#     """Preprocess the image for NuLite model."""
#
#     # binary_mask = output["nuclei_binary_map"].softmax(0).detach().cpu().numpy()[1]
#     # hv_map = output["hv_map"].detach().cpu().numpy()
#
#     _, blb = cv2.threshold(
#         nuclei_binary_map.astype(np.float32), 0.5, 1, cv2.THRESH_BINARY
#     )
#     blb = blb.astype(np.uint8)
#
#     # Remove small objects based on connected components.
#     # Use cv2.connectedComponentsWithStats to label regions and filter by area.
#     num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(blb, connectivity=8)
#     min_size = 3  # Minimum pixel area to keep an object
#     blb_clean = np.zeros_like(blb)
#     for label in range(1, num_labels):  # label 0 is the background.
#         if stats[label, cv2.CC_STAT_AREA] >= min_size:
#             blb_clean[labels == label] = 1
#
#     h_map, v_map = hv_map
#
#     # STEP 2: Normalize directional maps
#     h_dir_norm = cv2.normalize(
#         h_map, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX
#     ).astype(np.float32)
#     v_dir_norm = cv2.normalize(
#         v_map, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX
#     ).astype(np.float32)
#
#     # STEP 3: Compute edges using Sobel operators
#     # ksize = 11  # Kernel size for Sobel operators; adjust for edge sensitivity.
#     sobelh = cv2.Sobel(h_dir_norm, cv2.CV_64F, dx=1, dy=0, ksize=ksize)
#     sobelv = cv2.Sobel(v_dir_norm, cv2.CV_64F, dx=0, dy=1, ksize=ksize)
#
#     # Normalize the edge responses and invert them to prepare for the "distance" map.
#     sobelh_norm = 1 - cv2.normalize(
#         sobelh, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX
#     )
#     sobelv_norm = 1 - cv2.normalize(
#         sobelv, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX
#     )
#
#     # Combine edge images by taking the maximum value at each pixel.
#     overall = np.maximum(sobelh_norm, sobelv_norm)
#
#     # Remove non-nuclei regions from the edge map.
#     overall = overall - (1 - blb_clean.astype(np.float32))
#     overall[overall < 0] = 0  # Set negative values to zero
#
#     # STEP 5: Create markers for watershed (seed regions)
#     # Identify the nucleus interior by thresholding the overall edge image.
#     _, overall_bin = cv2.threshold(overall, 0.4, 1, cv2.THRESH_BINARY)
#     overall_bin = overall_bin.astype(np.uint8)
#
#     # Subtract the boundaries from the clean binary mask
#     marker = blb_clean - overall_bin
#     marker[marker < 0] = 0
#
#     # Fill holes and do a morphological closing to smooth marker regions.
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#     marker_closed = cv2.morphologyEx(marker, cv2.MORPH_CLOSE, kernel)
#
#     # Again, remove tiny markers using connected component analysis.
#     num_labels, markers, stats, _ = cv2.connectedComponentsWithStats(
#         marker_closed, connectivity=8
#     )
#     object_size = 10  # Minimum size (in pixels) for a marker region
#     markers_clean = np.zeros_like(markers, dtype=np.int32)
#     for label in range(1, num_labels):
#         if stats[label, cv2.CC_STAT_AREA] >= object_size:
#             markers_clean[markers == label] = label
#
#     # STEP 6: Apply the Watershed algorithm using only OpenCV
#     # The watershed function in OpenCV requires a 3-channel image.
#     # Here, we build a dummy 3-channel (RGB) image from our binary mask (for visualization/masking purposes).
#     dummy_img = cv2.cvtColor((blb_clean * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
#
#     # Watershed modifies the marker image in place.
#     # The boundaries between segmented regions will be marked with -1.
#     cv2.watershed(dummy_img, markers_clean)
#     return markers_clean


def np_hv_postprocess(
    nuclei_binary_map: np.ndarray,
    hv_map: np.ndarray,
    variant: str = "20x",  # "20x" or "40x"
):
    blb_raw = nuclei_binary_map
    hv_raw = hv_map

    h_dir_raw = hv_raw[0]
    v_dir_raw = hv_raw[1]

    # hyperparameters
    if variant == "20x":
        min_size, kernel_size = 5, 11
    else:
        min_size, kernel_size = 10, 21

    # processing
    blb = np.array(blb_raw >= 0.5, dtype=np.int32)
    blb = label(blb)[0]
    blb = remove_small_objects(blb, min_size=min_size)
    blb[blb > 0] = 1  # background is 0 already

    h_dir = cv2.normalize(  # type: ignore
        h_dir_raw,
        None,
        alpha=0,
        beta=1,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_32F,
    )
    v_dir = cv2.normalize(  # type: ignore
        v_dir_raw,
        None,
        alpha=0,
        beta=1,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_32F,
    )

    sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=kernel_size)

    sobelh = 1 - (
        cv2.normalize(  # type: ignore
            sobelh,
            None,
            alpha=0,
            beta=1,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,
        )
    )
    sobelv = 1 - (
        cv2.normalize(  # type: ignore
            sobelv,
            None,
            alpha=0,
            beta=1,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,
        )
    )

    overall = np.maximum(sobelh, sobelv)
    overall = overall - (1 - blb)
    overall[overall < 0] = 0

    dist = (1.0 - overall) * blb
    ## nuclei values form mountains so inverse to get basins
    dist = -cv2.GaussianBlur(dist, (3, 3), 0)

    overall = np.array(overall >= 0.4, dtype=np.int32)

    marker = blb - overall
    marker[marker < 0] = 0
    marker = binary_fill_holes(marker).astype("uint8")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
    marker = label(marker)[0]
    marker = remove_small_objects(marker, min_size=min_size)

    proced_pred = watershed(dist, markers=marker, mask=blb)

    return proced_pred


def remove_small_objects(
    pred: np.ndarray, min_size: int = 64, connectivity: int = 1
) -> np.ndarray:
    """Remove connected components smaller than the specified size.

    This function is taken from skimage.morphology.remove_small_objects, but the warning
    is removed when a single label is provided.

    Taken from: https://github.com/vqdang/hover_net/blob/master/misc/utils.py#L142

    Parameters
    ----------
    pred: np.ndarray
        Input array.
    min_size: int
        Minimum size of instance in output array
    connectivity: int
        The connectivity defining the neighborhood of a pixel.

    Returns
    -------
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

    component_sizes = np.bincount(ccs.ravel())

    too_small = component_sizes < min_size
    too_small_mask = too_small[ccs]
    out[too_small_mask] = 0

    return out
