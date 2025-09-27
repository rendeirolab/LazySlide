from dataclasses import dataclass
from typing import Optional

import geopandas as gpd
import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class SegmentationStats:
    """Statistics for segmentation evaluation.

    A data class containing confusion matrix elements and optional IoU values
    for evaluating segmentation performance.

    Attributes
    ----------
    tp : int
        True positives - correctly identified positive instances.
    fp : int
        False positives - incorrectly identified positive instances.
    fn : int
        False negatives - missed positive instances.
    tn : int
        True negatives - correctly identified negative instances.
    ious : list of float, optional
        List of intersection over union (IoU) values for matched instances.
        Only populated for instance segmentation tasks. Default is None.
    """

    tp: int
    fp: int
    fn: int
    tn: int
    ious: Optional[list[float]] = None


def get_instance_stats(
    gdf_true: gpd.GeoDataFrame,
    gdf_pred: gpd.GeoDataFrame,
    iou_threshold: float = 0.5,
) -> SegmentationStats:
    """Compute instance segmentation statistics using Hungarian matching.

    This function evaluates instance segmentation performance by matching
    predicted instances to ground truth instances using optimal assignment
    based on intersection over union (IoU) scores. Uses the Hungarian algorithm
    to find the optimal one-to-one matching that maximizes total IoU.

    Parameters
    ----------
    gdf_true : gpd.GeoDataFrame
        Ground truth instances as a GeoDataFrame with geometry column containing
        polygon geometries representing true object instances.
    gdf_pred : gpd.GeoDataFrame
        Predicted instances as a GeoDataFrame with geometry column containing
        polygon geometries representing predicted object instances.
    iou_threshold : float, default=0.5
        Minimum IoU threshold for considering a match as a true positive.
        Matches with IoU below this threshold are considered false positives.

    Returns
    -------
    SegmentationStats
        Statistics object containing:
        - tp: Number of true positive matches (IoU >= threshold)
        - fp: Number of false positive predictions (unmatched predictions)
        - fn: Number of false negative ground truths (unmatched ground truths)
        - tn: Always 0 (not meaningful for instance segmentation)
        - ious: List of IoU values for all valid matches
    """
    n_true, n_pred = len(gdf_true), len(gdf_pred)
    iou_matrix = np.zeros((n_true, n_pred))

    # Build spatial index for predictions
    sindex = gdf_pred.sindex

    for i, gt in enumerate(gdf_true.geometry):
        # Candidate matches using bounding-box overlap
        candidate_ids = list(sindex.intersection(gt.bounds))
        for j in candidate_ids:
            pr = gdf_pred.geometry.iloc[j]
            inter = gt.intersection(pr).area
            if inter > 0:
                union = gt.area + pr.area - inter
                iou_matrix[i, j] = inter / union

    # Hungarian matching (maximize IoU)
    cost_matrix = -iou_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = []
    for r, c in zip(row_ind, col_ind):
        if iou_matrix[r, c] >= iou_threshold:
            matches.append(iou_matrix[r, c])

    tp = len(matches)
    fn = n_true - tp
    fp = n_pred - tp
    tn = 0  # TN not meaningful for polygons

    return SegmentationStats(tp=tp, fp=fp, fn=fn, tn=tn, ious=matches)


def get_semantic_stats(
    gdf_true: gpd.GeoDataFrame,
    gdf_pred: gpd.GeoDataFrame,
) -> SegmentationStats:
    """Compute semantic segmentation statistics using area-based overlap.

    This function evaluates semantic segmentation performance by computing
    area-based statistics between ground truth and predicted polygons.
    It calculates intersection areas to determine true positives, false
    positives, and false negatives based on geometric overlap.

    Parameters
    ----------
    gdf_true : gpd.GeoDataFrame
        Ground truth semantic segmentation as a GeoDataFrame with geometry
        column containing polygon geometries representing true regions.
    gdf_pred : gpd.GeoDataFrame
        Predicted semantic segmentation as a GeoDataFrame with geometry
        column containing polygon geometries representing predicted regions.

    Returns
    -------
    SegmentationStats
        Statistics object containing:
        - tp: True positive area (intersection between ground truth and predictions)
        - fp: False positive area (predicted area not overlapping with ground truth)
        - fn: False negative area (ground truth area not covered by predictions)
        - tn: Always 0 (background true negatives not well-defined for polygons)
        - ious: Always None (not applicable for semantic segmentation)
    """
    tp = fp = fn = tn = 0

    gt_c = gdf_true.unary_union
    pr_c = gdf_pred.unary_union

    inter = gt_c.intersection(pr_c).area
    # union = gt_c.union(pr_c).area

    tp_c = inter
    fp_c = pr_c.area - inter
    fn_c = gt_c.area - inter
    tn_c = 0  # for polygons, background TN not well-defined

    tp += tp_c
    fp += fp_c
    fn += fn_c
    tn += tn_c

    return SegmentationStats(tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))


def accuracy(stats: SegmentationStats) -> float:
    """Compute accuracy from segmentation statistics.

    Accuracy is the proportion of correctly classified instances (both positive
    and negative) out of the total instances.

    Parameters
    ----------
    stats : SegmentationStats
        Segmentation statistics containing true positives (tp), false positives (fp),
        false negatives (fn), and true negatives (tn).

    Returns
    -------
    float
        Accuracy score in the range [0, 1], where 1 is perfect accuracy.
        Returns 0.0 if the denominator is zero.

    Notes
    -----
    Accuracy = (TP + TN) / (TP + FP + FN + TN)
    """
    denom = stats.tp + stats.fp + stats.fn + stats.tn
    return (stats.tp + stats.tn) / denom if denom > 0 else 0.0


def precision(stats: SegmentationStats) -> float:
    """Compute precision from segmentation statistics.

    Precision measures the proportion of predicted positive instances that are
    actually positive. It answers the question: "Of all the instances we predicted
    as positive, how many were actually positive?"

    Parameters
    ----------
    stats : SegmentationStats
        Segmentation statistics containing true positives (tp) and false positives (fp).

    Returns
    -------
    float
        Precision score in the range [0, 1], where 1 is perfect precision.
        A small epsilon (1e-8) is added to the denominator to avoid division by zero.

    Notes
    -----
    Precision = TP / (TP + FP)
    """
    return stats.tp / (stats.tp + stats.fp + 1e-8)


def recall(stats: SegmentationStats) -> float:
    """Compute recall (sensitivity) from segmentation statistics.

    Recall measures the proportion of actual positive instances that were correctly
    identified. It answers the question: "Of all the instances that are actually
    positive, how many did we correctly predict as positive?"

    Parameters
    ----------
    stats : SegmentationStats
        Segmentation statistics containing true positives (tp) and false negatives (fn).

    Returns
    -------
    float
        Recall score in the range [0, 1], where 1 is perfect recall.
        A small epsilon (1e-8) is added to the denominator to avoid division by zero.

    Notes
    -----
    Recall = TP / (TP + FN)
    Also known as sensitivity or true positive rate (TPR).
    """
    return stats.tp / (stats.tp + stats.fn + 1e-8)


def f1_score(stats: SegmentationStats) -> float:
    """Compute F1 score from segmentation statistics.

    The F1 score is the harmonic mean of precision and recall, providing a single
    metric that balances both precision and recall. It reaches its best value at 1
    (perfect precision and recall) and worst at 0.

    Parameters
    ----------
    stats : SegmentationStats
        Segmentation statistics containing true positives (tp), false positives (fp),
        and false negatives (fn) used to compute precision and recall.

    Returns
    -------
    float
        F1 score in the range [0, 1], where 1 indicates perfect F1 score.
        A small epsilon (1e-8) is added to the denominator to avoid division by zero.

    Notes
    -----
    F1 = 2 * (precision * recall) / (precision + recall)
    F1 = 2 * TP / (2 * TP + FP + FN)
    """
    p, r = precision(stats), recall(stats)
    return 2 * p * r / (p + r + 1e-8)


def mean_iou(stats: SegmentationStats) -> float:
    """Compute mean Intersection over Union (mIoU) from segmentation statistics.

    This function handles both instance and semantic segmentation scenarios:
    - For instance segmentation: computes the mean of IoU values for matched instances
    - For semantic segmentation: computes IoU using the Jaccard index formula

    Parameters
    ----------
    stats : SegmentationStats
        Segmentation statistics. If stats.ious is not None (instance segmentation),
        uses the list of IoU values. Otherwise (semantic segmentation), uses
        tp, fp, and fn counts to compute IoU.

    Returns
    -------
    float
        Mean IoU score in the range [0, 1], where 1 indicates perfect overlap.
        For instance segmentation: returns 0.0 if no matches exist.
        For semantic segmentation: a small epsilon (1e-8) prevents division by zero.

    Notes
    -----
    Instance segmentation: mIoU = mean(IoU_values)
    Semantic segmentation: IoU = TP / (TP + FP + FN)
    """
    if stats.ious is not None:  # instance seg
        return np.mean(stats.ious) if stats.ious else 0.0
    else:
        denom = stats.tp + stats.fp + stats.fn
        return stats.tp / (denom + 1e-8)


def dice(stats: SegmentationStats) -> float:
    """Compute Dice coefficient from segmentation statistics.

    The Dice coefficient (also known as Sørensen-Dice coefficient or F1 score)
    measures the similarity between two sets by computing twice the size of their
    intersection divided by the total size of both sets.

    Parameters
    ----------
    stats : SegmentationStats
        Segmentation statistics containing true positives (tp), false positives (fp),
        and false negatives (fn).

    Returns
    -------
    float
        Dice coefficient in the range [0, 1], where 1 indicates perfect overlap.
        A small epsilon (1e-8) is added to the denominator to avoid division by zero.

    Notes
    -----
    Dice = 2 * TP / (2 * TP + FP + FN)
    This is mathematically equivalent to the F1 score.
    """
    return 2 * stats.tp / (2 * stats.tp + stats.fp + stats.fn + 1e-8)


def sensitivity(stats: SegmentationStats) -> float:
    """Compute sensitivity (true positive rate) from segmentation statistics.

    Sensitivity measures the proportion of actual positive instances that were
    correctly identified. This is identical to recall and represents the ability
    of the model to correctly identify positive cases.

    Parameters
    ----------
    stats : SegmentationStats
        Segmentation statistics containing true positives (tp) and false negatives (fn).

    Returns
    -------
    float
        Sensitivity score in the range [0, 1], where 1 indicates perfect sensitivity.
        This function delegates to the recall function for computation.

    Notes
    -----
    Sensitivity = TP / (TP + FN)
    Sensitivity is mathematically identical to recall and true positive rate (TPR).

    See Also
    --------
    recall : Equivalent function with the same computation.
    """
    return recall(stats)


def specificity(stats: SegmentationStats) -> float:
    """Compute specificity (true negative rate) from segmentation statistics.

    Specificity measures the proportion of actual negative instances that were
    correctly identified as negative. It represents the ability of the model
    to correctly identify negative cases and avoid false positives.

    Parameters
    ----------
    stats : SegmentationStats
        Segmentation statistics containing true negatives (tn) and false positives (fp).

    Returns
    -------
    float
        Specificity score in the range [0, 1], where 1 indicates perfect specificity.
        A small epsilon (1e-8) is added to the denominator to avoid division by zero.

    Notes
    -----
    Specificity = TN / (TN + FP)
    Also known as true negative rate (TNR).
    Specificity is the complement of the false positive rate: Specificity = 1 - FPR.
    """
    denom = stats.tn + stats.fp
    return stats.tn / (denom + 1e-8)


def pq(stats: SegmentationStats) -> float:
    """Compute Panoptic Quality (PQ) from segmentation statistics.

    Panoptic Quality is a unified metric for evaluating panoptic segmentation
    that combines both detection quality (matching instances) and segmentation
    quality (IoU of matched instances). It balances both recognition and
    segmentation quality in a single metric.

    Parameters
    ----------
    stats : SegmentationStats
        Segmentation statistics containing IoU values for matched instances,
        true positives (tp), false positives (fp), and false negatives (fn).
        The ious field must not be None for meaningful PQ computation.

    Returns
    -------
    float
        Panoptic Quality score in the range [0, 1], where 1 indicates perfect
        panoptic segmentation quality. Returns 0.0 if no IoU values are available
        or if there are no true positive matches.

    Notes
    -----
    PQ = (sum of IoUs for matched pairs) / (TP + 0.5 * FP + 0.5 * FN)

    The denominator weights unmatched instances (FP and FN) at half the weight
    of matched instances (TP) to balance detection and segmentation quality.
    """
    if stats.ious is None or stats.tp == 0:
        return 0.0
    return sum(stats.ious) / (stats.tp + 0.5 * stats.fp + 0.5 * stats.fn)
