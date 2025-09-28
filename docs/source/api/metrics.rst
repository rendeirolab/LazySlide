Metrics
-------


Segmentation
~~~~~~~~~~~~

LazySlide provides comprehensive segmentation evaluation metrics for both instance and semantic segmentation tasks.

Here's how to evaluate segmentation performance using LazySlide's metrics:

.. code-block:: python

    import geopandas as gpd
    from shapely.geometry import Polygon
    from lazyslide.metrics.segmentation import (
        get_instance_stats,
        get_semantic_stats,
        accuracy,
        mean_iou,
        pq,
    )

    # Ground truth in geodataframe
    gt_polygons = [
        Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),  # Square 1
        Polygon([(15, 15), (25, 15), (25, 25), (15, 25)])  # Square 2
    ]
    gdf_true = gpd.GeoDataFrame({'geometry': gt_polygons})

    # Prediction in geodataframe
    pred_polygons = [
        Polygon([(1, 1), (9, 1), (9, 9), (1, 9)]),      # Slightly smaller square 1
        Polygon([(16, 16), (24, 16), (24, 24), (16, 24)]),  # Slightly smaller square 2
        Polygon([(30, 30), (40, 30), (40, 40), (30, 40)])   # False positive
    ]
    gdf_pred = gpd.GeoDataFrame({'geometry': pred_polygons})

    # Instance segmentation evaluation
    instance_stats = get_instance_stats(gdf_true, gdf_pred, iou_threshold=0.5)

    # Semantic segmentation evaluation
    semantic_stats = get_semantic_stats(gdf_true, gdf_pred)

    # Calculate various metrics
    acc = accuracy(semantic_stats)
    miou = mean_iou(instance_stats)
    pq = pq(instance_stats)

.. currentmodule:: lazyslide.metrics.segmentation

.. autosummary::
    :toctree: _autogen
    :nosignatures:

    SegmentationStats
    get_instance_stats
    get_semantic_stats
    accuracy
    precision
    recall
    f1_score
    mean_iou
    dice
    sensitivity
    specificity
    pq


TopK
~~~~

.. currentmodule:: lazyslide.metrics

.. autosummary::
    :toctree: _autogen
    :nosignatures:

    topk


