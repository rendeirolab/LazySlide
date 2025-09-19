import numpy as np
import pytest
import torch

from lazyslide.models.base import StyleTransferModel
from lazyslide.models.style_transfer import ROSIE

# Import get_token to check if user is logged in to HuggingFace
try:
    from huggingface_hub.utils._auth import get_token

    HF_TOKEN_AVAILABLE = get_token() is not None
except ImportError:
    HF_TOKEN_AVAILABLE = False


# Define a function to conditionally skip tests if HF token is not available
def skip_if_no_token(*model_class):
    """Skip test if no HuggingFace token is available for gated models."""
    if not HF_TOKEN_AVAILABLE:
        return pytest.param(
            *model_class, marks=pytest.mark.skip(reason="Requires HF token")
        )
    return pytest.param(*model_class)


# List of style transfer models to test
STYLE_TRANSFER_MODELS = [
    skip_if_no_token(ROSIE),
]


def run_style_transfer_model_tests(model_class):
    """Run all tests for a style transfer model.

    This function runs all tests for a single style transfer model, then releases the model
    to free memory before returning. This ensures that only one model is in memory
    at a time, which is important for large models that can consume a lot of memory.

    Args:
        model_class: The model class to test
    """
    # Initialize the model
    model = model_class()

    # Test 1: Model initialization
    assert isinstance(model, StyleTransferModel)
    assert hasattr(model, "model")
    assert hasattr(model, "predict")

    # Test 2: Model attributes
    assert hasattr(model, "task")
    assert hasattr(model, "description")
    assert hasattr(model, "license")
    assert hasattr(model, "commercial")

    # Test 3: Get transform
    transform = model.get_transform()
    assert transform is not None
    assert callable(transform)

    # Test 4: To device
    model_on_device = model.to("cpu")
    assert model_on_device is model  # Should return self

    # Check device
    torch_model = model.model
    if hasattr(torch_model, "device"):
        # Handle DataParallel models
        if hasattr(torch_model, "module"):
            # DataParallel case
            pass  # DataParallel doesn't have a simple device attribute
        else:
            assert torch_model.device.type == "cpu"

    # Create test images - ROSIE expects 224x224 RGB images
    mock_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    # Test 5: Transform image
    # Apply transform to numpy image
    transformed_image = transform(mock_image)
    assert isinstance(transformed_image, torch.Tensor)
    assert transformed_image.shape == (3, 224, 224)  # CHW format after transform

    # Add batch dimension for prediction
    image_tensor = transformed_image.unsqueeze(0)

    # Test 6: Model prediction
    with torch.inference_mode():
        output = model.predict(image_tensor)

    # Check output shape and type
    assert isinstance(output, torch.Tensor)
    assert output.shape[0] == 1  # batch size
    assert output.shape[1] == 50  # ROSIE outputs 50 channels
    assert torch.is_floating_point(output)

    # Test 7: Batch prediction
    batch_tensor = torch.stack(
        [transformed_image, transformed_image]
    )  # Create batch of 2
    with torch.inference_mode():
        batch_output = model.predict(batch_tensor)

    assert isinstance(batch_output, torch.Tensor)
    assert batch_output.shape[0] == 2  # batch size
    assert batch_output.shape[1] == 50  # ROSIE outputs 50 channels
    assert torch.is_floating_point(batch_output)

    # Test 9: Input tile check (ROSIE-specific)
    if hasattr(model, "check_input_tile"):
        # Test with recommended mpp
        result = model.check_input_tile(mpp=0.25, size_x=224, size_y=224)
        assert isinstance(result, bool)

    # Explicitly delete the model to free memory
    del model


# Test function for all style transfer models
@pytest.mark.gpu
@pytest.mark.parametrize("model_class", STYLE_TRANSFER_MODELS)
def test_style_transfer_model(model_class):
    """Test a style transfer model.

    This test function runs all tests for a single style transfer model, then releases the model
    to free memory before moving to the next model. This ensures that only one model
    is in memory at a time, which is important for large models that can consume a lot
    of memory.
    """
    # Run all tests for this model
    run_style_transfer_model_tests(model_class)
