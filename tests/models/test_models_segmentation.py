from numbers import Integral

import numpy as np
import pytest
import torch

from lazyslide.models.base import SegmentationModel

# Import all segmentation models
from lazyslide.models.segmentation import (
    SAM,
    Cellpose,
    GrandQCArtifact,
    GrandQCTissue,
    Instanseg,
    NuLite,
    PathProfilerTissueSegmentation,
)

# Define the models to test
CELL_SEGMENTATION_MODELS = [
    NuLite,
    Cellpose,
    Instanseg,
]

SEMANTIC_SEGMENTATION_MODELS = [
    SAM,
    GrandQCTissue,
    GrandQCArtifact,
    PathProfilerTissueSegmentation,
]

ALL_SEGMENTATION_MODELS = CELL_SEGMENTATION_MODELS + SEMANTIC_SEGMENTATION_MODELS


def run_segmentation_model_tests(model_class):
    """Run all tests for a segmentation model.

    This function runs all tests for a single segmentation model, then releases the model
    to free memory before returning. This ensures that only one model is in memory
    at a time, which is important for large models that can consume a lot of memory.

    Args:
        model_class: The model class to test
    """
    # Initialize the model
    model = model_class()

    # Test 1: Model initialization
    assert isinstance(model, SegmentationModel)
    # Check for model attribute or any attribute that contains "model"
    has_model_attr = hasattr(model, "model")
    if not has_model_attr:
        # Check for alternative model attributes
        for attr in dir(model):
            if "model" in attr.lower() and not attr.startswith("__"):
                has_model_attr = True
                break
    assert has_model_attr, (
        "Model does not have a model attribute or any attribute containing 'model'"
    )
    assert hasattr(model, "segment")
    assert hasattr(model, "supported_output")

    # Test 2: Get transform
    transform = model.get_transform()
    if transform is not None:
        assert callable(transform)

    # Test 3: To device
    model_on_device = model.to("cpu")
    # Some models might not return self from to()
    if model_on_device is not None:
        assert model_on_device is model  # Should return self

    # Check device - handle different model attributes and JIT models
    # Find the model attribute
    model_attr = None
    if hasattr(model, "model"):
        model_attr = model.model
    else:
        for attr in dir(model):
            if "model" in attr.lower() and not attr.startswith("__"):
                model_attr = getattr(model, attr)
                break

    # Check device if the model attribute has a device
    if model_attr is not None and hasattr(model_attr, "device"):
        assert model_attr.device.type == "cpu"

    # Create test images
    mock_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    # Apply transform if available
    if transform is not None:
        image_tensor = transform(mock_image).unsqueeze(0)
    else:
        # If no transform, create an uint8 image tensor
        image_tensor = torch.from_numpy(mock_image).unsqueeze(0)

    # Check supported outputs
    supported_outputs = model.supported_output()
    assert isinstance(supported_outputs, (list, tuple))

    allowed_supported_outputs = {
        "probability_map",
        "instance_map",
        "class_map",
        "token_map",
    }
    for s in supported_outputs:
        assert s in allowed_supported_outputs

    # Test 4: Segment
    segmentation = model.segment(image_tensor)

    # Check that the output is valid
    assert segmentation is not None
    assert isinstance(segmentation, dict)  # Must return a dictionary

    for key in segmentation:
        assert key in supported_outputs
        vs = segmentation[key]
        assert isinstance(vs, (torch.Tensor, np.ndarray))

        if key == "probability_map":
            # Should be in range [0, 1]
            assert torch.is_tensor(vs)
            vmin = vs.min()
            vmax = vs.max()
            assert 0 <= vmin <= 1, f"Minimum value {vmin} is out of range [0, 1]"
            assert 0 <= vmax <= 1, f"Maximum value {vmax} is out of range [0, 1]"
        elif key == "instance_map":
            # Must be integer dtype
            if isinstance(vs, torch.Tensor):
                assert vs.dtype in {
                    torch.uint8,
                    torch.uint16,
                    torch.uint32,
                    torch.uint64,
                    torch.int8,
                    torch.int16,
                    torch.int32,
                    torch.int64,
                }
        # TODO: test class_map and token_map if they are supported

    # Explicitly delete the model to free memory
    del model


# Test function for all segmentation models
@pytest.mark.gpu
@pytest.mark.parametrize("model_class", ALL_SEGMENTATION_MODELS)
def test_segmentation_model(model_class):
    """Test a segmentation model.

    This test function runs all tests for a single segmentation model, then releases the model
    to free memory before moving to the next model. This ensures that only one model
    is in memory at a time, which is important for large models that can consume a lot
    of memory.
    """
    # Run all tests for this model
    run_segmentation_model_tests(model_class)
