import numpy as np
from anndata import AnnData


def topk_score(
    matrix: np.ndarray | AnnData,
    k: int = 5,
    agg_method: str = "max",
) -> np.ndarray:
    """
    Get the top k score from a feature x class matrix.

    Parameters
    ----------
    matrix : np.ndarray | AnnData
        The input matrix. Feature x class.
    k : int, default: 5
        The number of top scores to return.
    agg_method : str, default: "max"
        The method to use for aggregation.
        Can be "max", "mean", "median" or "sum".

    Returns
    -------
    np.ndarray
        The top k scores.

    """
    if isinstance(matrix, AnnData):
        matrix = matrix.X

    top_k_score = np.sort(matrix, axis=0)[-k:]
    score = getattr(np, agg_method)(top_k_score, axis=0)
    return score
