from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np

from ...cv.scorer.utils import dtype_limits
from ..base import ModelTask, TilePredictionModel

__all__ = [
    "SplitRGB",
    "Brightness",
    "Contrast",
    "Sharpness",
    "Sobel",
    "Canny",
    "Entropy",
    "Saturation",
    "HaralickTexture",
]


def _correct_image_format(image):
    # Convert from BCHW to HWC format for processing
    if image.shape[0] == 3:  # CHW format
        image = np.transpose(image, (1, 2, 0))  # Convert to HWC
    return image


class _CVFeatures(TilePredictionModel, ABC, abstract=True):
    task = ModelTask.cv_feature
    commercial = False
    license = None

    def to(self, device):
        pass

    def get_transform(self):
        return None

    @abstractmethod
    def _func(self, image):
        """
        Abstract method to be implemented by subclasses.
        This method should process a single image and return the feature.
        """
        pass

    def _process_batch(self, images):
        results = []
        # Process each image in the batch
        for image in images:
            image = _correct_image_format(image)
            results.append(self._func(image))

        if isinstance(results[0], dict):
            batch_results = {
                key: np.array([r[key] for r in results]) for key in results[0].keys()
            }
            return batch_results
        else:
            return {self.key: np.array(results)}

    def predict(self, image):
        image = np.asarray(image)
        if image.ndim == 3:
            # Batch it
            image = np.expand_dims(image, 0)
        return self._process_batch(image)


class CVCompose(_CVFeatures, abstract=True):
    """
    Compose multiple CV features into a single feature.

    This class allows you to combine multiple CV features into a single feature.
    It is useful for creating a composite feature that includes multiple aspects
    of the image, such as brightness, contrast, and color information.

    Parameters
    ----------
    *models : list of _CVFeatures
        List of CV feature instances to be composed.
    """

    def __init__(self, *models):
        self.models = models

    def _func(self, image):
        pass

    def predict(self, image):
        image = np.asarray(image)
        if image.ndim == 3:
            # Batch it
            image = np.expand_dims(image, 0)

        results = {}
        for model in self.models:
            model_results = model.predict(image)
            results.update(model_results)
        return results


class SplitRGB(_CVFeatures, key="split_rgb"):
    """
    Calculate the RGB value of a tile.

    Brightness is calculated as the mean of the pixel values.

    Parameters
    ----------
    method : str
        Method to calculate the RGB value. Default is "mean".
    dim : str
        Dimension of the image. Default is "xyc".
    """

    def __init__(
        self,
        method: str = "mean",
    ):
        self.method = method

    def _func(self, image):
        c_int = getattr(image, self.method)(axis=(0, 1))
        return {"red": c_int[0], "green": c_int[1], "blue": c_int[2]}


class Brightness(_CVFeatures, key="brightness"):
    """
    Calculate the brightness of a tile.

    The tile can be in shape (H, W, C) for a single image or (B, C, H, W) for a batch of images.
    """

    def _func(self, image):
        return image.mean()


class Contrast(_CVFeatures, key="contrast"):
    """
    Calculate the contrast of a tile.

    Contrast is calculated as the standard deviation of the pixel values.

    The tile can be in shape (H, W, C) for a single image or (B, C, H, W) for a batch of images.

    Parameters
    ----------
    lower_percentile : float
        Lower percentile for contrast calculation.
    upper_percentile : float
        Upper percentile for contrast calculation.
    """

    def __init__(
        self,
        lower_percentile: float = 1,
        upper_percentile: float = 99,
    ):
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

    def _func(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        dlimits = dtype_limits(image, clip_negative=False)
        limits = np.percentile(image, [self.lower_percentile, self.upper_percentile])
        ratio = (limits[1] - limits[0]) / (dlimits[1] - dlimits[0])
        return ratio


class Sharpness(_CVFeatures, key="sharpness"):
    """
    Calculate the sharpness of a tile.

    Sharpness is calculated as the variance of the Laplacian of the pixel values.
    The Laplacian operator is used to measure the second derivative of an image,
    which highlights regions of rapid intensity change and is therefore often used
    for edge detection. High variance in the Laplacian indicates a sharper image.

    The tile can be in shape (H, W, C) for a single image or (B, C, H, W) for a batch of images.
    """

    def _func(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Apply Laplacian operator
        laplacian = cv2.Laplacian(gray_image.astype(np.float32), cv2.CV_32F)
        return laplacian.var()


class Sobel(_CVFeatures, key="sobel"):
    """
    Calculate the sobel of a tile.

    Sobel is calculated as the variance of the Sobel of the pixel values.

    The Sobel operator calculates the gradient of the image intensity at each pixel,
    giving the direction of the largest possible increase from light to dark and the
    rate of change in that direction.

    The tile can be in shape (H, W, C) for a single image or (B, C, H, W) for a batch of images.
    """

    def __init__(self, ksize=3):
        self.ksize = ksize

    def _func(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Calculate Sobel in x and y directions
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=self.ksize)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=self.ksize)

        # Calculate the magnitude of gradients
        magnitude = np.sqrt(sobelx**2 + sobely**2)

        # Calculate variance of the magnitude
        return magnitude.var()


class Canny(_CVFeatures, key="canny"):
    """
    Calculate the canny edge detection score of a tile.

    The Canny edge detector is an edge detection operator that uses a multi-stage
    algorithm to detect a wide range of edges in images. The score is calculated
    as the variance of the edge-detected image.

    The tile can be in shape (H, W, C) for a single image or (B, C, H, W) for a batch of images.

    Parameters
    ----------
    low_threshold : int
        Lower threshold for the hysteresis procedure in Canny edge detection.
    high_threshold : int
        Higher threshold for the hysteresis procedure in Canny edge detection.
    """

    def __init__(self, low_threshold=100, high_threshold=200):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def _func(self, image):
        # Convert to grayscale if the image is in color
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Apply Canny edge detection
        edges = cv2.Canny(
            gray_image.astype(np.uint8), self.low_threshold, self.high_threshold
        )

        # Calculate variance of the edge-detected image
        return edges.var()


class Entropy(_CVFeatures, key="entropy"):
    """
    Calculate the entropy of a tile.

    Entropy is a statistical measure of randomness that can be used to characterize
    the texture of an image. Higher entropy indicates more complex textures and
    potentially more information content.

    The tile can be in shape (H, W, C) for a single image or (B, C, H, W) for a batch of images.
    """

    def _func(self, image):
        # Convert to grayscale if the image is in color
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Calculate histogram
        hist = cv2.calcHist([gray_image.astype(np.uint8)], [0], None, [256], [0, 256])

        # Normalize histogram to get probability distribution
        hist = hist / hist.sum()

        # Remove zero probabilities (log(0) is undefined)
        hist = hist[hist > 0]

        # Calculate entropy
        entropy_value = -np.sum(hist * np.log2(hist))

        return entropy_value


class Saturation(_CVFeatures, key="saturation"):
    """
    Calculate the color saturation of a tile.

    Saturation measures the colorfulness of an image. It is calculated by converting
    the image to HSV color space and taking the mean of the saturation channel.
    Higher values indicate more vibrant colors.

    The tile can be in shape (H, W, C) for a single image or (B, C, H, W) for a batch of images.
    """

    def _func(self, image):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Extract saturation channel (second channel in HSV)
        saturation_channel = hsv_image[:, :, 1]

        # Calculate mean saturation
        mean_saturation = saturation_channel.mean() / 255.0  # Normalize to [0, 1]

        return mean_saturation


class HaralickTexture(_CVFeatures, key="haralick_texture"):
    """
    Calculate texture features using Gray Level Co-occurrence Matrix (GLCM).

    This class implements Haralick texture features which are derived from the GLCM.
    These features provide information about the texture of an image and are widely
    used in image analysis.

    The tile can be in shape (H, W, C) for a single image or (B, C, H, W) for a batch of images.

    Parameters
    ----------
    distances : list of int
        List of pixel pair distance offsets.
    angles : list of float
        List of pixel pair angles in radians.
    levels : int
        Number of gray levels to use in the GLCM.
    """

    def __init__(self, distances=None, angles=None, levels=8):
        self.distances = distances if distances is not None else [1]
        self.angles = (
            angles if angles is not None else [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        )
        self.levels = levels

    def _calculate_glcm(self, image):
        """Calculate the Gray Level Co-occurrence Matrix."""
        # Quantize the image to reduce the number of intensity values
        bins = np.linspace(0, 255, self.levels + 1)
        quantized = np.digitize(image, bins) - 1

        # Calculate GLCM for each distance and angle
        glcm = np.zeros(
            (self.levels, self.levels, len(self.distances), len(self.angles))
        )

        for i, distance in enumerate(self.distances):
            for j, angle in enumerate(self.angles):
                # Calculate offset
                dx = int(round(distance * np.cos(angle)))
                dy = int(round(distance * np.sin(angle)))

                # Create shifted image
                rows, cols = quantized.shape
                shifted = np.zeros_like(quantized)

                if dx >= 0:
                    col_range = range(0, cols - dx)
                    shifted_col_range = range(dx, cols)
                else:
                    col_range = range(-dx, cols)
                    shifted_col_range = range(0, cols + dx)

                if dy >= 0:
                    row_range = range(0, rows - dy)
                    shifted_row_range = range(dy, rows)
                else:
                    row_range = range(-dy, rows)
                    shifted_row_range = range(0, rows + dy)

                shifted[
                    shifted_row_range[0] : shifted_row_range[-1] + 1,
                    shifted_col_range[0] : shifted_col_range[-1] + 1,
                ] = quantized[
                    row_range[0] : row_range[-1] + 1, col_range[0] : col_range[-1] + 1
                ]

                # Calculate co-occurrence matrix
                for k in range(self.levels):
                    for level in range(self.levels):
                        glcm[k, level, i, j] = np.sum(
                            (quantized == k) & (shifted == level)
                        )

                # Normalize GLCM
                if glcm[:, :, i, j].sum() > 0:
                    glcm[:, :, i, j] /= glcm[:, :, i, j].sum()

        return glcm

    def _calculate_haralick_features(self, glcm):
        """Calculate Haralick features from GLCM."""
        features = {}

        # Average over all GLCMs
        mean_glcm = glcm.mean(axis=(2, 3))

        # Calculate features
        # 1. Energy (Angular Second Moment)
        features["energy"] = np.sum(mean_glcm**2)

        # 2. Contrast
        indices = np.arange(self.levels)
        i, j = np.meshgrid(indices, indices)
        features["contrast"] = np.sum(mean_glcm * ((i - j) ** 2))

        # 3. Homogeneity (Inverse Difference Moment)
        features["homogeneity"] = np.sum(mean_glcm / (1 + (i - j) ** 2))

        # 4. Correlation
        pi = mean_glcm.sum(axis=1)
        pj = mean_glcm.sum(axis=0)

        mu_i = np.sum(indices * pi)
        mu_j = np.sum(indices * pj)

        sigma_i = np.sqrt(np.sum(pi * ((indices - mu_i) ** 2)))
        sigma_j = np.sqrt(np.sum(pj * ((indices - mu_j) ** 2)))

        if sigma_i > 0 and sigma_j > 0:
            corr_num = np.sum(mean_glcm * np.outer(indices - mu_i, indices - mu_j))
            features["correlation"] = corr_num / (sigma_i * sigma_j)
        else:
            features["correlation"] = 0

        # 5. Entropy
        non_zero = mean_glcm > 0
        if np.any(non_zero):
            features["entropy"] = -np.sum(
                mean_glcm[non_zero] * np.log2(mean_glcm[non_zero])
            )
        else:
            features["entropy"] = 0

        return features

    def _func(self, image):
        # Convert to grayscale if the image is in color
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Ensure image is uint8
        if gray_image.dtype != np.uint8:
            gray_image = (
                (gray_image * 255).astype(np.uint8)
                if gray_image.max() <= 1
                else gray_image.astype(np.uint8)
            )

        # Calculate GLCM
        glcm = self._calculate_glcm(gray_image)

        # Calculate Haralick features
        features = self._calculate_haralick_features(glcm)

        # Prefix feature names with 'texture_'
        scores = {f"texture_{k}": v for k, v in features.items()}

        return scores
