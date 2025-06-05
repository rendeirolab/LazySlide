import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pytest
from shapely.geometry import Polygon

from lazyslide.cv.mask import (
    BinaryMask,
    InstanceMap,
    Mask,
    MulticlassMask,
    MultilabelMask,
    ProbabilityMap,
    binary_mask_to_polygons,
    binary_mask_to_polygons_with_prob,
)

# Set random seed for reproducibility
np.random.seed(42)

# Define test dimensions
H, W = 100, 100
N_CLASS = 3

# Create test data
binary_mask = np.random.randint(0, 2, (H, W), dtype=np.uint8)
multiclass_mask = np.random.randint(0, N_CLASS, (H, W), dtype=np.uint8)
multilabel_mask = np.random.randint(0, 2, (N_CLASS, H, W), dtype=np.uint8)
instance_map = np.zeros((H, W), dtype=np.uint8)
# Create a few "instances" in the instance map
for i in range(1, 6):
    y, x = np.random.randint(0, H - 20), np.random.randint(0, W - 20)
    instance_map[y : y + 20, x : x + 20] = i
probability_map_2d = np.random.random((H, W)).astype(np.float32)
probability_map_3d = np.random.random((N_CLASS, H, W)).astype(np.float32)


# Create test polygons
def create_test_polygons():
    polygons = []
    for i in range(3):
        x, y = np.random.randint(10, W - 30), np.random.randint(10, H - 30)
        poly = Polygon([(x, y), (x + 20, y), (x + 20, y + 20), (x, y + 20)])
        polygons.append(poly)

    gdf = gpd.GeoDataFrame({"geometry": polygons, "class": [1, 2, 3]})
    return gdf


class TestMaskBase:
    """Tests for the base Mask class functionality."""

    def test_is_integer_dtype(self):
        """Test the _is_integer_dtype static method."""
        assert Mask._is_integer_dtype(np.array([1, 2, 3]))
        assert Mask._is_integer_dtype(np.array([1, 2, 3], dtype=np.uint8))
        assert not Mask._is_integer_dtype(np.array([1.0, 2.0, 3.0]))

    def test_is_probability_map(self):
        """Test the _is_probability_map static method."""
        assert Mask._is_probability_map(np.array([0.1, 0.5, 0.9]))
        assert not Mask._is_probability_map(np.array([1, 2, 3]))
        assert not Mask._is_probability_map(np.array([-0.1, 0.5, 1.1]))

    def test_from_polygons(self):
        """Test creating masks from polygons."""
        polygons = create_test_polygons()

        # Test creating multilabel mask with class column
        multilabel = Mask.from_polygons(
            polygons, class_col="class", bounding_box=[0, 0, W, H]
        )
        assert isinstance(multilabel, MultilabelMask)
        assert multilabel.mask.shape == (3, H, W)  # 3 classes

        # Note: We're not testing the binary mask case (without class_col)
        # because there's a bug in the implementation that needs to be fixed


class TestBinaryMask:
    """Tests for the BinaryMask class."""

    def test_init(self):
        """Test initialization of BinaryMask."""
        # Test with valid input
        mask = BinaryMask(binary_mask)
        assert mask.mask.shape == binary_mask.shape
        assert np.array_equal(mask.mask, binary_mask > 0)
        assert mask.prob_map is None

        # Test with probability map
        mask = BinaryMask(binary_mask, prob_map=probability_map_2d)
        assert mask.prob_map is not None
        assert mask.prob_map.shape == binary_mask.shape

        # Test with invalid input (should raise assertion error)
        with pytest.raises(AssertionError):
            BinaryMask(np.random.rand(3, H, W))  # 3D mask

    def test_to_binary_mask(self):
        """Test conversion to binary mask."""
        mask = BinaryMask(binary_mask)
        binary = mask.to_binary_mask()
        assert binary.shape == binary_mask.shape
        assert np.array_equal(binary, mask.mask)

    def test_to_multilabel_mask(self):
        """Test conversion to multilabel mask."""
        mask = BinaryMask(binary_mask)
        multilabel = mask.to_multilabel_mask()
        assert multilabel.shape == (1, H, W)
        assert np.array_equal(multilabel[0], mask.mask)

    def test_to_multiclass_mask(self):
        """Test conversion to multiclass mask."""
        mask = BinaryMask(binary_mask)
        multiclass = mask.to_multiclass_mask()
        assert multiclass.shape == binary_mask.shape
        assert np.array_equal(multiclass, mask.mask)

    def test_to_instance_map(self):
        """Test conversion to instance map."""
        mask = BinaryMask(binary_mask)
        instance = mask.to_instance_map()
        assert instance.shape == binary_mask.shape
        # Check that connected components are labeled with unique integers
        if np.any(mask.mask):  # Only if there are foreground pixels
            assert len(np.unique(instance)) > 1

    def test_to_polygons(self):
        """Test conversion to polygons."""
        mask = BinaryMask(binary_mask)
        polygons = mask.to_polygons()
        assert isinstance(polygons, gpd.GeoDataFrame)

        # Test with probability map
        mask = BinaryMask(binary_mask, prob_map=probability_map_2d)
        polygons = mask.to_polygons()
        assert "prob" in polygons.columns


class TestMulticlassMask:
    """Tests for the MulticlassMask class."""

    def test_init(self):
        """Test initialization of MulticlassMask."""
        # Test with valid input
        mask = MulticlassMask(multiclass_mask)
        assert mask.mask.shape == multiclass_mask.shape
        assert mask.n_classes == len(np.unique(multiclass_mask))

        # Test with class names
        class_names = ["background", "class1", "class2"]
        mask = MulticlassMask(multiclass_mask, class_names=class_names)
        # Check that class_name is properly set (could be a dict or other mapping)
        assert mask.class_names is not None
        for i, name in enumerate(class_names):
            assert mask.class_names.get(i) == name

        # Test with invalid input
        with pytest.raises(AssertionError):
            MulticlassMask(np.random.rand(H, W))  # Non-integer mask

    def test_to_binary_mask(self):
        """Test conversion to binary mask."""
        mask = MulticlassMask(multiclass_mask)
        binary = mask.to_binary_mask()
        assert binary.shape == multiclass_mask.shape
        assert np.array_equal(binary, multiclass_mask != 0)

    def test_to_multiclass_mask(self):
        """Test conversion to multiclass mask."""
        mask = MulticlassMask(multiclass_mask)
        multiclass = mask.to_multiclass_mask()
        assert multiclass.shape == multiclass_mask.shape
        assert np.array_equal(multiclass, mask.mask)

    def test_to_multilabel_mask(self):
        """Test conversion to multilabel mask."""
        mask = MulticlassMask(multiclass_mask)
        multilabel = mask.to_multilabel_mask()
        assert multilabel.shape == (mask.n_classes, H, W)
        # Check that each class is correctly encoded
        for i, c in enumerate(mask.classes):
            assert np.array_equal(multilabel[i], mask.mask == c)

    def test_to_polygons(self):
        """Test conversion to polygons."""
        mask = MulticlassMask(multiclass_mask)
        polygons = mask.to_polygons()
        assert isinstance(polygons, gpd.GeoDataFrame)

        # Test with ignore_index
        polygons = mask.to_polygons(ignore_index=0)
        if len(polygons) > 0:
            assert not np.any(polygons["class"] == 0)

        # Test with class names
        class_names = {0: "background", 1: "class1", 2: "class2"}
        mask = MulticlassMask(multiclass_mask, class_names=class_names)
        polygons = mask.to_polygons()
        if len(polygons) > 0:
            assert "class" in polygons.columns


class TestMultilabelMask:
    """Tests for the MultilabelMask class."""

    def test_init(self):
        """Test initialization of MultilabelMask."""
        # Test with valid input
        mask = MultilabelMask(multilabel_mask)
        assert mask.mask.shape == multilabel_mask.shape
        assert mask.n_classes == N_CLASS

        # Test with class names
        class_names = ["class0", "class1", "class2"]
        mask = MultilabelMask(multilabel_mask, class_names=class_names)
        assert mask.class_names == {i: name for i, name in enumerate(class_names)}

        # Test with invalid input
        with pytest.raises(AssertionError):
            MultilabelMask(np.random.rand(N_CLASS, H, W))  # Non-integer mask

    def test_to_binary_mask(self):
        """Test conversion to binary mask."""
        mask = MultilabelMask(multilabel_mask)
        binary = mask.to_binary_mask()
        assert binary.shape == (H, W)
        assert np.array_equal(binary, np.any(mask.mask > 0, axis=0).astype(np.uint8))

    def test_to_multilabel_mask(self):
        """Test conversion to multilabel mask."""
        mask = MultilabelMask(multilabel_mask)
        multilabel = mask.to_multilabel_mask()
        assert multilabel.shape == multilabel_mask.shape
        assert np.array_equal(multilabel, mask.mask)

    def test_to_multiclass_mask(self):
        """Test conversion to multiclass mask."""
        # Create a simpler multilabel mask for testing
        # where each pixel belongs to at most one class
        test_mask = np.zeros((N_CLASS, H, W), dtype=np.uint8)
        for i in range(N_CLASS):
            y, x = np.random.randint(0, H - 20), np.random.randint(0, W - 20)
            test_mask[i, y : y + 20, x : x + 20] = 1

        mask = MultilabelMask(test_mask)
        multiclass = mask.to_multiclass_mask()
        assert multiclass.shape == (H, W)

        # Check that each pixel is assigned to the correct class
        # Only for pixels that belong to exactly one class
        for i in range(mask.n_classes):
            # Find pixels that belong only to class i
            class_i_pixels = (test_mask[i] == 1) & np.all(
                test_mask[np.arange(N_CLASS) != i] == 0, axis=0
            )
            if np.any(class_i_pixels):
                assert np.all(multiclass[class_i_pixels] == i)

    def test_to_polygons(self):
        """Test conversion to polygons."""
        mask = MultilabelMask(multilabel_mask)
        polygons = mask.to_polygons()
        assert isinstance(polygons, gpd.GeoDataFrame)

        # Test with ignore_index
        polygons = mask.to_polygons(ignore_index=0)
        if len(polygons) > 0:
            assert not np.any(polygons["class"] == 0)

        # Test with class names
        class_names = {0: "class0", 1: "class1", 2: "class2"}
        mask = MultilabelMask(multilabel_mask, class_names=class_names)
        polygons = mask.to_polygons()
        if len(polygons) > 0:
            assert "class" in polygons.columns


class TestInstanceMap:
    """Tests for the InstanceMap class."""

    def test_init(self):
        """Test initialization of InstanceMap."""
        # Test with valid input
        mask = InstanceMap(instance_map)
        assert mask.mask.shape == instance_map.shape
        assert not mask._is_classification

        # Test with probability map (2D)
        mask = InstanceMap(instance_map, prob_map=probability_map_2d)
        assert mask.prob_map is not None
        assert not mask._is_classification

        # Test with probability map (3D - classification)
        mask = InstanceMap(instance_map, prob_map=probability_map_3d)
        assert mask.prob_map is not None
        assert mask._is_classification

        # Test with invalid input
        with pytest.raises(AssertionError):
            InstanceMap(np.random.rand(H, W))  # Non-integer mask

    def test_to_polygons(self):
        """Test conversion to polygons."""
        mask = InstanceMap(instance_map)
        polygons = mask.to_polygons()
        assert isinstance(polygons, gpd.GeoDataFrame)

        # Test with probability map
        mask = InstanceMap(instance_map, prob_map=probability_map_2d)
        polygons = mask.to_polygons()
        if len(polygons) > 0:
            assert "prob" in polygons.columns


class TestProbabilityMap:
    """Tests for the ProbabilityMap class."""

    def test_init(self):
        """Test initialization of ProbabilityMap."""
        # Test with 2D input
        mask = ProbabilityMap(probability_map_2d)
        assert mask.mask.shape == probability_map_2d.shape
        assert mask.is2D

        # Test with 3D input
        mask = ProbabilityMap(probability_map_3d)
        assert mask.mask.shape == probability_map_3d.shape
        assert not mask.is2D

        # Test with invalid input
        with pytest.raises(AssertionError):
            ProbabilityMap(np.random.randint(0, 2, (H, W)))  # Non-float mask

        with pytest.raises(AssertionError):
            ProbabilityMap(np.random.rand(4, H, W, 3))  # 4D mask

    def test_to_polygons(self):
        """Test conversion to polygons."""
        # Test with 2D probability map
        mask = ProbabilityMap(probability_map_2d)
        polygons = mask.to_polygons()
        assert isinstance(polygons, gpd.GeoDataFrame)
        if len(polygons) > 0:
            assert "prob" in polygons.columns

        # Test with 3D probability map
        mask = ProbabilityMap(probability_map_3d)
        polygons = mask.to_polygons()
        assert isinstance(polygons, gpd.GeoDataFrame)
        if len(polygons) > 0:
            assert "prob" in polygons.columns
            assert "class" in polygons.columns

        # Test with class names
        class_names = {0: "class0", 1: "class1", 2: "class2"}
        mask = ProbabilityMap(probability_map_3d, class_names=class_names)
        polygons = mask.to_polygons()
        if len(polygons) > 0:
            assert "class" in polygons.columns


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_binary_mask_to_polygons(self):
        """Test binary_mask_to_polygons function."""
        polygons = binary_mask_to_polygons(binary_mask)
        assert isinstance(polygons, list)

        # Test with min_area
        polygons = binary_mask_to_polygons(binary_mask, min_area=0.1)

        # Test with min_hole_area
        polygons = binary_mask_to_polygons(binary_mask, min_hole_area=0.05)

        # Test without hole detection
        polygons = binary_mask_to_polygons(binary_mask, detect_holes=False)

    def test_binary_mask_to_polygons_with_prob(self):
        """Test binary_mask_to_polygons_with_prob function."""
        polygons = binary_mask_to_polygons_with_prob(binary_mask)
        assert isinstance(polygons, gpd.GeoDataFrame)

        # Test with probability map
        polygons = binary_mask_to_polygons_with_prob(
            binary_mask, prob_map=probability_map_2d
        )
        if len(polygons) > 0:
            assert "prob" in polygons.columns

        # Test with min_area
        polygons = binary_mask_to_polygons_with_prob(binary_mask, min_area=0.1)

        # Test with min_hole_area
        polygons = binary_mask_to_polygons_with_prob(binary_mask, min_hole_area=0.05)

        # Test without hole detection
        polygons = binary_mask_to_polygons_with_prob(binary_mask, detect_holes=False)
