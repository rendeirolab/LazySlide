from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union

import cv2
import geopandas as gpd
import numpy as np
import torch
from shapely import Polygon
from shapely.affinity import scale
from skimage.filters import threshold_otsu
from wsidata import WSIData
from wsidata.io import add_shapes

from lazyslide._utils import default_pbar, get_torch_device
from lazyslide.cv import BinaryMask
from lazyslide.models.segmentation import SAM

# Configure logging
logger = logging.getLogger(__name__)


def _initialize_model(
    model_name: str, model_kwargs: Optional[Dict] = None, device: Optional[str] = None
) -> Tuple[object, str]:
    """
    Initialize the segmentation model.

    Parameters
    ----------
    model_name : str
        The name of the model to use.
    model_kwargs : dict, optional
        Additional keyword arguments for the model.
    device : str, optional
        The device to run the model on (e.g., 'cpu', 'cuda').

    Returns
    -------
    Tuple[object, str]
        The initialized model instance and the device it's running on.
    """
    if model_name == "sam":
        model_instance = SAM(**(model_kwargs or {}))
    else:
        raise ValueError(
            f"Unsupported model: {model_name}. Currently only 'sam' is supported."
        )

    if device is None:
        device = get_torch_device()

    model_instance.to(device)
    logger.info(f"Model initialized on device: {device}")

    return model_instance, device


def _get_threshold_value(
    threshold_param: Union[str, float, List[float]], values: np.ndarray, index: int = 0
) -> float:
    """
    Determine the threshold value based on the provided parameter.

    Parameters
    ----------
    threshold_param : str | float | list[float]
        Thresholding method or value for segmentation.
    values : np.ndarray
        The values to threshold.
    index : int, default: 0
        The index to use if threshold_param is a list.

    Returns
    -------
    float
        The threshold value.
    """
    if threshold_param == "otsu":
        if len(values) == 0:
            raise ValueError("Cannot compute Otsu threshold on empty array")
        return threshold_otsu(values)
    elif isinstance(threshold_param, (float, int)):
        return float(threshold_param)
    elif isinstance(threshold_param, list):
        if index >= len(threshold_param):
            raise IndexError(
                f"Threshold list index {index} out of range (list length: {len(threshold_param)})"
            )
        return threshold_param[index]
    else:
        raise ValueError(
            f"Threshold must be 'otsu', a float, or a list of floats, got {type(threshold_param)}"
        )


def _create_tile_mask(
    mask_shape: Tuple[int, int],
    tile_points: np.ndarray,
    tile_height: int,
    tile_width: int,
    kernel_size: int = 7,
    sigma: int = 5,
) -> np.ndarray:
    """
    Create and process a mask for the tiles.

    Parameters
    ----------
    mask_shape : tuple[int, int]
        The shape of the mask (height, width).
    tile_points : np.ndarray
        The points representing tile centers.
    tile_height : int
        The height of each tile.
    tile_width : int
        The width of each tile.
    kernel_size : int, default: 7
        The size of the kernel for morphological operations.
    sigma : int, default: 5
        The sigma value for Gaussian blur.

    Returns
    -------
    np.ndarray
        The processed tile mask.
    """
    # Create initial mask
    tile_mask = np.full(mask_shape, 0, dtype=np.uint8)

    # Fill mask with tiles
    for x, y in tile_points:
        # Ensure coordinates are within bounds
        if 0 <= y < mask_shape[0] and 0 <= x < mask_shape[1]:
            y_end = min(y + tile_height + 1, mask_shape[0])
            x_end = min(x + tile_width + 1, mask_shape[1])
            tile_mask[y:y_end, x:x_end] = 1

    # Process mask
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    # Apply Gaussian blur
    tile_mask = cv2.GaussianBlur(tile_mask, (kernel_size, kernel_size), sigma, sigma)

    # Apply morphological operations
    tile_mask = cv2.morphologyEx(
        src=tile_mask, kernel=kernel, op=cv2.MORPH_OPEN, iterations=1
    )
    tile_mask = cv2.morphologyEx(
        src=tile_mask, kernel=kernel, op=cv2.MORPH_CLOSE, iterations=1
    )

    return tile_mask


def _process_final_mask(
    mask: np.ndarray, kernel_size: int = 7, sigma: int = 5
) -> np.ndarray:
    """
    Process the final segmentation mask.

    Parameters
    ----------
    mask : np.ndarray
        The input mask to process.
    kernel_size : int, default: 7
        The size of the kernel for morphological operations.
    sigma : int, default: 5
        The sigma value for Gaussian blur.

    Returns
    -------
    np.ndarray
        The processed mask.
    """
    # Convert to binary mask
    binary_mask = mask.squeeze().astype(np.bool_).astype(np.uint8)

    # Apply Gaussian blur
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    smoothed_mask = cv2.GaussianBlur(
        binary_mask, (kernel_size, kernel_size), sigma, sigma
    )

    # Apply morphological operations
    smoothed_mask = cv2.morphologyEx(
        src=smoothed_mask, kernel=kernel, op=cv2.MORPH_OPEN, iterations=1
    )
    smoothed_mask = cv2.morphologyEx(
        src=smoothed_mask, kernel=kernel, op=cv2.MORPH_CLOSE, iterations=1
    )

    return smoothed_mask


def _segment_with_model(
    model_instance, image, embeddings, pos_prompts, neg_prompts, boxes
) -> np.ndarray:
    """
    Perform segmentation using the model.

    Parameters
    ----------
    model_instance : object
        The segmentation model instance.
    image : np.ndarray
        The image to segment.
    embeddings : torch.Tensor
        The image embeddings.
    pos_prompts : list
        Positive prompt points.
    neg_prompts : list
        Negative prompt points.
    boxes : list
        Bounding boxes for segmentation.

    Returns
    -------
    np.ndarray
        The segmentation mask.
    """
    try:
        # Combine positive and negative prompts
        all_prompts = pos_prompts + neg_prompts
        all_labels = [1] * len(pos_prompts) + [0] * len(neg_prompts)

        # Perform segmentation
        mask = model_instance.segment(
            image,
            image_embedding=embeddings,
            input_points=[all_prompts],
            input_labels=[all_labels],
            input_boxes=[boxes],
        )

        return mask
    except Exception as e:
        logger.error(f"Error during segmentation: {str(e)}")
        # Return empty mask in case of error
        return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)


def zero_shot(
    wsi: WSIData,
    prompts: List[str],
    table_key: str,
    tile_key: str,
    tissue_key: str = "tissues",
    threshold: Union[str, float, List[float]] = "otsu",
    model: str = "sam",
    device: Optional[str] = None,
    model_kwargs: Optional[Dict] = None,
    key_added: str = "zero_shot_segmentation",
    min_area: float = 10,
    show_progress: bool = True,
) -> WSIData:
    """
    Perform zero-shot segmentation on the WSI using the specified model and prompts.

    Parameters
    ----------
    wsi : :class:`WSIData <wsidata.WSIData>`
        The WSIData object to work on.
    prompts : list[str]
        List of prompts for zero-shot segmentation.
    table_key : str
        The key for the similarity matrix in the WSIData object.
    tile_key : str
        The key for the tiles in the WSIData object.
    tissue_key : str, default: "tissues"
        The key for the tissues in the WSIData object.
    threshold : str | float | list[float], default: 'otsu'
        Thresholding method or value for segmentation.
        - 'otsu': Use Otsu's method to determine threshold
        - float: Use a fixed threshold value
        - list[float]: Use different threshold values for each prompt
    model : str, default: "sam"
        The model to use for segmentation.
    device : str, optional
        The device to run the model on (e.g., 'cpu', 'cuda').
    model_kwargs : dict, optional
        Additional keyword arguments for the model.
    key_added : str, default: "zero_shot_segmentation"
        The key to store the results in the WSIData object.
    min_area : float, default: 10
        Minimum area for polygons to be included in the results.
    show_progress : bool, default: True
        Whether to show a progress bar.

    Returns
    -------
    WSIData
        The updated WSIData object with segmentation results.
    """
    # Input validation
    if not prompts:
        raise ValueError("Prompts list cannot be empty")

    if not isinstance(wsi, WSIData):
        raise TypeError(f"Expected WSIData object, got {type(wsi)}")

    if table_key not in wsi:
        raise KeyError(f"Table key '{table_key}' not found in WSIData object")

    if tile_key not in wsi:
        raise KeyError(f"Tile key '{tile_key}' not found in WSIData object")

    if tissue_key not in wsi:
        raise KeyError(f"Tissue key '{tissue_key}' not found in WSIData object")

    # Initialize model
    model_instance, device = _initialize_model(model, model_kwargs, device)

    # Initialize results list
    segment_results = []

    # Get similarity matrix and tile specification
    similarity_matrix = wsi[table_key]
    spec = wsi.tile_spec(tile_key)

    # Get tissues for processing
    tissues = list(wsi.iter.tissue_images(tissue_key, level=-1))

    # Set up progress bar if requested
    with default_pbar(disable=not show_progress) as pbar:
        tissue_task = pbar.add_task("Processing tissues", total=len(tissues))

        # Process each tissue
        for d in tissues:
            # Get the slice of the current tissue
            cut = wsi[tile_key]["tissue_id"] == d.tissue_id

            # Skip if no tiles for this tissue
            if not cut.any():
                logger.warning(f"No tiles found for tissue ID {d.tissue_id}")
                pbar.update(tissue_task, advance=1)
                continue

            # Calculate dimensions
            mask_shape = (d.height, d.width)
            tile_height = int(spec.base_height / d.downsample)
            tile_width = int(spec.base_width / d.downsample)

            # Scale tissue contour to original coordinates
            tissue_contour = scale(
                d.contour, xfact=d.downsample, yfact=d.downsample, origin=(0, 0)
            )
            bounds = tissue_contour.bounds
            xoff, yoff = bounds[0], bounds[1]

            # Transform tiles to tissue coordinate system
            tiles = (
                wsi[tile_key]
                .loc[cut]
                .geometry.translate(xoff=-xoff, yoff=-yoff)
                .scale(xfact=1 / d.downsample, yfact=1 / d.downsample, origin=(0, 0))
            )

            # Get image embeddings
            try:
                embeddings = model_instance.get_image_embedding(d.image)
            except Exception as e:
                logger.error(
                    f"Error getting image embeddings for tissue ID {d.tissue_id}: {str(e)}"
                )
                pbar.update(tissue_task, advance=1)
                continue

            # Add prompt task to progress bar
            prompt_task = pbar.add_task(
                f"Processing prompts for tissue {d.tissue_id}", total=len(prompts)
            )

            # Process each prompt
            for ix, prompt in enumerate(prompts):
                try:
                    # Get similarity values
                    vs = similarity_matrix[cut, prompt].X.flatten()

                    # Skip if no similarity values
                    if len(vs) == 0:
                        logger.warning(
                            f"No similarity values for prompt '{prompt}' in tissue ID {d.tissue_id}"
                        )
                        pbar.update(prompt_task, advance=1)
                        continue

                    # Get threshold value
                    try:
                        thresh = _get_threshold_value(threshold, vs, ix)
                    except Exception as e:
                        logger.error(
                            f"Error determining threshold for prompt '{prompt}': {str(e)}"
                        )
                        pbar.update(prompt_task, advance=1)
                        continue

                    # Separate positive and negative tiles
                    pos_tiles = tiles[vs > thresh]
                    neg_tiles = tiles[vs < thresh]

                    # Skip if no positive tiles
                    if len(pos_tiles) == 0:
                        logger.warning(
                            f"No positive tiles for prompt '{prompt}' in tissue ID {d.tissue_id}"
                        )
                        pbar.update(prompt_task, advance=1)
                        continue

                    # Get tile points
                    tile_pts = (
                        pos_tiles.geometry.centroid.get_coordinates().values.astype(int)
                    )

                    # Create and process tile mask
                    tile_mask = _create_tile_mask(
                        mask_shape, tile_pts, tile_height, tile_width
                    )

                    # Find contours
                    contours, _ = cv2.findContours(
                        tile_mask.copy(),
                        mode=cv2.RETR_EXTERNAL,
                        method=cv2.CHAIN_APPROX_NONE,
                    )

                    # Skip if no contours
                    if not contours:
                        logger.warning(
                            f"No contours found for prompt '{prompt}' in tissue ID {d.tissue_id}"
                        )
                        pbar.update(prompt_task, advance=1)
                        continue

                    # Create polygons from contours
                    try:
                        polys = gpd.GeoSeries(
                            [Polygon(cnt.squeeze()) for cnt in contours if cnt.size > 0]
                        )
                    except Exception as e:
                        logger.error(f"Error creating polygons from contours: {str(e)}")
                        pbar.update(prompt_task, advance=1)
                        continue

                    # Process each polygon
                    results = []
                    for poly in polys:
                        # Get bounding box
                        boxes = np.array([poly.bounds]).astype(int).tolist()

                        # Get prompts within polygon
                        pos_prompts = (
                            pos_tiles[pos_tiles.intersects(poly)]
                            .centroid.get_coordinates()
                            .astype(int)
                            .values.tolist()
                        )
                        neg_prompts = (
                            neg_tiles[neg_tiles.intersects(poly)]
                            .centroid.get_coordinates()
                            .astype(int)
                            .values.tolist()
                        )

                        # Skip if no positive prompts
                        if not pos_prompts:
                            continue

                        # Perform segmentation
                        mask = _segment_with_model(
                            model_instance,
                            d.image,
                            embeddings,
                            pos_prompts,
                            neg_prompts,
                            boxes,
                        )
                        results.append(mask)

                    # Skip if no results
                    if not results:
                        logger.warning(
                            f"No segmentation results for prompt '{prompt}' in tissue ID {d.tissue_id}"
                        )
                        pbar.update(prompt_task, advance=1)
                        continue

                    # Combine and process results
                    final_mask = (
                        np.sum(results, axis=0)
                        .squeeze()
                        .astype(np.bool_)
                        .astype(np.uint8)
                    )
                    final_mask = _process_final_mask(final_mask)

                    # Convert mask to polygons
                    objects = BinaryMask(final_mask).to_polygons(min_area=min_area)

                    # Skip if no objects
                    if not objects:
                        logger.warning(
                            f"No objects found for prompt '{prompt}' in tissue ID {d.tissue_id}"
                        )
                        pbar.update(prompt_task, advance=1)
                        continue

                    # Transform polygons to original coordinate system
                    objects = gpd.GeoSeries(objects)
                    objects = objects.scale(
                        xfact=d.downsample, yfact=d.downsample, origin=(0, 0)
                    ).translate(xoff=xoff, yoff=yoff)

                    # Filter objects outside tissue contour
                    objects = objects[objects.intersects(tissue_contour)]

                    # Add objects to results
                    for obj in objects:
                        segment_results.append(
                            [
                                d.tissue_id,
                                prompt,
                                obj,
                            ]
                        )

                except Exception as e:
                    logger.error(
                        f"Error processing prompt '{prompt}' for tissue ID {d.tissue_id}: {str(e)}"
                    )

                pbar.update(prompt_task, advance=1)

            pbar.update(tissue_task, advance=1)

    # Create GeoDataFrame from results
    if not segment_results:
        logger.warning("No segmentation results found")
        results_df = gpd.GeoDataFrame(
            [],
            columns=["tissue_id", "prompt", "geometry"],
        )
    else:
        results_df = gpd.GeoDataFrame(
            segment_results,
            columns=["tissue_id", "prompt", "geometry"],
        )

    # Add results to WSIData object
    add_shapes(wsi, key_added, results_df)

    logger.info(f"Zero-shot segmentation completed. Results stored in '{key_added}'")
    return wsi
