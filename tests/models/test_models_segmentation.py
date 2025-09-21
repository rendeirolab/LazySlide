import numpy as np
import pytest
import torch
from huggingface_hub.errors import GatedRepoError

from lazyslide.models import MODEL_REGISTRY, list_models
from lazyslide.models.base import SegmentationModel

# Define the models to test
SEGMENTATION_MODELS = list_models(task="segmentation")


@pytest.mark.large_runner
@pytest.mark.parametrize("model_name", SEGMENTATION_MODELS)
def test_segmentation_model(model_name):
    """Run all tests for a segmentation model.

    This function runs all tests for a single segmentation model, then releases the model
    to free memory before returning. This ensures that only one model is in memory
    at a time, which is important for large models that can consume a lot of memory.

    Args:
        model_class: The model class to test
    """
    # Initialize the model
    try:
        model = MODEL_REGISTRY[model_name]()
    except GatedRepoError:
        pytest.skip(f"{model_name} is not available.")
        return

    # Test 1: Model initialization
    assert isinstance(model, SegmentationModel)
    assert hasattr(model, "segment")
    assert hasattr(model, "supported_output")

    # Check supported_output
    SUPPORTED_OUTPUTS = {
        "probability_map",
        "instance_map",
        "class_map",
        "token_map",
    }
    assert set(model.supported_output()).issubset(SUPPORTED_OUTPUTS)

    # Create test images
    mock_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    # Apply transform if available
    transform = model.get_transform()
    if transform is not None:
        image_tensor = transform(mock_image).unsqueeze(0)
    else:
        # If no transform, create an uint8 image tensor
        image_tensor = torch.from_numpy(mock_image).unsqueeze(0)

    # Segment
    segmentation = model.segment(image_tensor)

    # Check that the output is valid
    assert segmentation is not None
    assert isinstance(segmentation, dict)  # Must return a dictionary

    for key in segmentation:
        assert key in SUPPORTED_OUTPUTS
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
