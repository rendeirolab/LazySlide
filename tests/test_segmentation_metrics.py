import numpy as np
import pytest

from lazyslide.metrics.segmentation import (
    SegmentationStats,
    accuracy,
    dice,
    f1_score,
    mean_iou,
    pq,
    precision,
    recall,
    sensitivity,
    specificity,
)


class TestSegmentationStats:
    """Test the SegmentationStats dataclass."""

    def test_segmentation_stats_creation(self):
        """Test basic creation of SegmentationStats."""
        stats = SegmentationStats(tp=10, fp=5, fn=3, tn=20)
        assert stats.tp == 10
        assert stats.fp == 5
        assert stats.fn == 3
        assert stats.tn == 20
        assert stats.ious is None

    def test_segmentation_stats_with_ious(self):
        """Test SegmentationStats with IoU values."""
        ious = [0.8, 0.9, 0.7]
        stats = SegmentationStats(tp=3, fp=2, fn=1, tn=0, ious=ious)
        assert stats.ious == ious
        assert len(stats.ious) == 3


class TestSegmentationMetrics:
    """Test all segmentation metric functions."""

    @pytest.fixture
    def basic_stats(self):
        """Basic stats for testing."""
        return SegmentationStats(tp=10, fp=5, fn=3, tn=20)

    @pytest.fixture
    def perfect_stats(self):
        """Perfect classification stats (no errors)."""
        return SegmentationStats(tp=10, fp=0, fn=0, tn=20)

    @pytest.fixture
    def worst_stats(self):
        """Worst case stats (all wrong)."""
        return SegmentationStats(tp=0, fp=10, fn=10, tn=0)

    @pytest.fixture
    def zero_stats(self):
        """All zero stats."""
        return SegmentationStats(tp=0, fp=0, fn=0, tn=0)

    @pytest.fixture
    def stats_with_ious(self):
        """Stats with IoU values for instance segmentation."""
        return SegmentationStats(tp=3, fp=2, fn=1, tn=0, ious=[0.8, 0.9, 0.7])

    def test_accuracy(self, basic_stats, perfect_stats, worst_stats, zero_stats):
        """Test accuracy calculation."""
        # Basic case
        expected = (10 + 20) / (10 + 5 + 3 + 20)  # (TP + TN) / total
        assert accuracy(basic_stats) == pytest.approx(expected, rel=1e-6)

        # Perfect case
        assert accuracy(perfect_stats) == 1.0

        # Worst case
        assert accuracy(worst_stats) == 0.0

        # Zero case
        assert accuracy(zero_stats) == 0.0

    def test_precision(self, basic_stats, perfect_stats, worst_stats, zero_stats):
        """Test precision calculation."""
        # Basic case
        expected = 10 / (10 + 5)  # TP / (TP + FP)
        assert precision(basic_stats) == pytest.approx(expected, rel=1e-6)

        # Perfect case
        assert precision(perfect_stats) == pytest.approx(1.0, rel=1e-6)

        # Worst case (should handle division by zero)
        result = precision(worst_stats)
        assert result == pytest.approx(0.0, abs=1e-6)

        # Zero case (should handle division by zero)
        result = precision(zero_stats)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_recall(self, basic_stats, perfect_stats, worst_stats, zero_stats):
        """Test recall calculation."""
        # Basic case
        expected = 10 / (10 + 3)  # TP / (TP + FN)
        assert recall(basic_stats) == pytest.approx(expected, rel=1e-6)

        # Perfect case
        assert recall(perfect_stats) == pytest.approx(1.0, rel=1e-6)

        # Worst case
        assert recall(worst_stats) == 0.0

        # Zero case (should handle division by zero)
        result = recall(zero_stats)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_sensitivity(self, basic_stats, perfect_stats, worst_stats, zero_stats):
        """Test sensitivity calculation (should be same as recall)."""
        # Sensitivity should be identical to recall
        assert sensitivity(basic_stats) == recall(basic_stats)
        assert sensitivity(perfect_stats) == recall(perfect_stats)
        assert sensitivity(worst_stats) == recall(worst_stats)
        assert sensitivity(zero_stats) == recall(zero_stats)

    def test_specificity(self, basic_stats, perfect_stats, worst_stats, zero_stats):
        """Test specificity calculation."""
        # Basic case
        expected = 20 / (20 + 5)  # TN / (TN + FP)
        assert specificity(basic_stats) == pytest.approx(expected, rel=1e-6)

        # Perfect case
        assert specificity(perfect_stats) == pytest.approx(1.0, rel=1e-6)

        # Worst case (should handle division by zero)
        result = specificity(worst_stats)
        assert result == pytest.approx(0.0, abs=1e-6)

        # Zero case (should handle division by zero)
        result = specificity(zero_stats)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_f1_score(self, basic_stats, perfect_stats, worst_stats, zero_stats):
        """Test F1 score calculation."""
        # Basic case
        p = precision(basic_stats)
        r = recall(basic_stats)
        expected = 2 * p * r / (p + r)
        assert f1_score(basic_stats) == pytest.approx(expected, rel=1e-6)

        # Perfect case
        assert f1_score(perfect_stats) == pytest.approx(1.0, rel=1e-6)

        # Worst case
        assert f1_score(worst_stats) == pytest.approx(0.0, abs=1e-6)

        # Zero case
        assert f1_score(zero_stats) == pytest.approx(0.0, abs=1e-6)

    def test_mean_iou_without_ious(
        self, basic_stats, perfect_stats, worst_stats, zero_stats
    ):
        """Test mean IoU calculation without IoU list (semantic segmentation)."""
        # Basic case
        expected = 10 / (10 + 5 + 3)  # TP / (TP + FP + FN)
        assert mean_iou(basic_stats) == pytest.approx(expected, rel=1e-6)

        # Perfect case
        expected_perfect = 10 / (10 + 0 + 0)  # Should be 1.0
        assert mean_iou(perfect_stats) == pytest.approx(expected_perfect, rel=1e-6)

        # Worst case
        assert mean_iou(worst_stats) == pytest.approx(0.0, abs=1e-6)

        # Zero case
        assert mean_iou(zero_stats) == pytest.approx(0.0, abs=1e-6)

    def test_mean_iou_with_ious(self, stats_with_ious):
        """Test mean IoU calculation with IoU list (instance segmentation)."""
        expected = np.mean([0.8, 0.9, 0.7])  # Mean of IoU values
        assert mean_iou(stats_with_ious) == pytest.approx(expected, rel=1e-6)

        # Test with empty IoU list
        empty_iou_stats = SegmentationStats(tp=0, fp=0, fn=0, tn=0, ious=[])
        assert mean_iou(empty_iou_stats) == 0.0

    def test_dice(self, basic_stats, perfect_stats, worst_stats, zero_stats):
        """Test Dice coefficient calculation."""
        # Basic case
        expected = 2 * 10 / (2 * 10 + 5 + 3)  # 2*TP / (2*TP + FP + FN)
        assert dice(basic_stats) == pytest.approx(expected, rel=1e-6)

        # Perfect case
        expected_perfect = 2 * 10 / (2 * 10 + 0 + 0)  # Should be 1.0
        assert dice(perfect_stats) == pytest.approx(expected_perfect, rel=1e-6)

        # Worst case
        assert dice(worst_stats) == pytest.approx(0.0, abs=1e-6)

        # Zero case
        assert dice(zero_stats) == pytest.approx(0.0, abs=1e-6)

    def test_pq_without_ious(self, basic_stats):
        """Test panoptic quality without IoU list."""
        # Should return 0.0 when ious is None
        assert pq(basic_stats) == 0.0

    def test_pq_with_ious(self, stats_with_ious):
        """Test panoptic quality with IoU list."""
        # PQ = sum(IoUs) / (TP + 0.5*FP + 0.5*FN)
        iou_sum = sum([0.8, 0.9, 0.7])  # 2.4
        expected = iou_sum / (3 + 0.5 * 2 + 0.5 * 1)  # 2.4 / 5.0 = 0.48
        assert pq(stats_with_ious) == pytest.approx(expected, rel=1e-6)

        # Test with zero TP
        zero_tp_stats = SegmentationStats(tp=0, fp=2, fn=1, tn=0, ious=[])
        assert pq(zero_tp_stats) == 0.0

    def test_edge_cases(self):
        """Test various edge cases for numerical stability."""
        # Test very small values
        small_stats = SegmentationStats(tp=1, fp=1, fn=1, tn=1)

        # All metrics should work without throwing exceptions
        assert accuracy(small_stats) > 0
        assert precision(small_stats) > 0
        assert recall(small_stats) > 0
        assert sensitivity(small_stats) > 0
        assert specificity(small_stats) > 0
        assert f1_score(small_stats) > 0
        assert mean_iou(small_stats) > 0
        assert dice(small_stats) > 0

        # Test large values
        large_stats = SegmentationStats(tp=1000000, fp=500000, fn=300000, tn=2000000)

        # All metrics should work without overflow
        assert 0 <= accuracy(large_stats) <= 1
        assert 0 <= precision(large_stats) <= 1
        assert 0 <= recall(large_stats) <= 1
        assert 0 <= sensitivity(large_stats) <= 1
        assert 0 <= specificity(large_stats) <= 1
        assert 0 <= f1_score(large_stats) <= 1
        assert 0 <= mean_iou(large_stats) <= 1
        assert 0 <= dice(large_stats) <= 1

    def test_mathematical_relationships(self, basic_stats):
        """Test mathematical relationships between metrics."""
        # Sensitivity should equal recall
        assert sensitivity(basic_stats) == recall(basic_stats)

        # F1 score should be harmonic mean of precision and recall
        p = precision(basic_stats)
        r = recall(basic_stats)
        f1 = f1_score(basic_stats)
        expected_f1 = 2 * p * r / (p + r)
        assert f1 == pytest.approx(expected_f1, rel=1e-6)

        # All metrics should be between 0 and 1
        assert 0 <= accuracy(basic_stats) <= 1
        assert 0 <= precision(basic_stats) <= 1
        assert 0 <= recall(basic_stats) <= 1
        assert 0 <= sensitivity(basic_stats) <= 1
        assert 0 <= specificity(basic_stats) <= 1
        assert 0 <= f1_score(basic_stats) <= 1
        assert 0 <= mean_iou(basic_stats) <= 1
        assert 0 <= dice(basic_stats) <= 1

    def test_metric_optimization_performance(self):
        """Test that metrics can handle large datasets efficiently."""
        # Create stats with large numbers to test performance
        large_stats = SegmentationStats(
            tp=1000000,
            fp=500000,
            fn=300000,
            tn=2000000,
            ious=[0.8] * 1000000,  # Large IoU list
        )

        # All metrics should complete quickly (this is mainly a regression test)
        import time

        start = time.time()
        accuracy(large_stats)
        precision(large_stats)
        recall(large_stats)
        sensitivity(large_stats)
        specificity(large_stats)
        f1_score(large_stats)
        dice(large_stats)
        pq(large_stats)
        end = time.time()

        # Should complete in reasonable time (less than 1 second)
        assert end - start < 1.0

        # mean_iou with large IoU list
        start = time.time()
        mean_iou(large_stats)
        end = time.time()

        # Should also complete quickly
        assert end - start < 1.0
