import numpy as np
import pytest
import torch

from lazyslide.models.base import ImageTextModel

# Import multimodal models that implement both encode_image and encode_text
from lazyslide.models.multimodal import CONCH, PLIP, OmiCLIP

# Import get_token to check if user is logged in to HuggingFace
try:
    from huggingface_hub.utils._auth import get_token

    HF_TOKEN_AVAILABLE = get_token() is not None
except ImportError:
    HF_TOKEN_AVAILABLE = False


# Define a function to conditionally skip tests if HF token is not available
def skip_if_no_token(*model_class):
    """Skip the test if HF token is not available."""
    if not HF_TOKEN_AVAILABLE:
        return pytest.param(
            *model_class, marks=pytest.mark.skip(reason="Requires HF token")
        )
    return pytest.param(*model_class)


# Define lists of models for parametrization
MULTIMODAL_MODELS = [
    skip_if_no_token(CONCH),
    skip_if_no_token(PLIP),
    skip_if_no_token(OmiCLIP),
]


def run_multimodal_model_tests(model_class):
    """Run all tests for a multimodal model.

    This function runs all tests for a single multimodal model, then releases the model
    to free memory before returning. This ensures that only one model is in memory
    at a time, which is important for large models that can consume a lot of memory.

    Args:
        model_class: The model class to test
    """
    # Initialize the model
    model = model_class()

    # Test 1: Model initialization
    assert isinstance(model, ImageTextModel)
    assert hasattr(model, "model")
    assert hasattr(model, "name")
    assert hasattr(model, "encode_image")
    assert hasattr(model, "encode_text")

    # Test 2: Get transform
    transform = model.get_transform()
    if transform is not None:
        assert callable(transform)

    # Test 3: To device
    model_on_device = model.to("cpu")
    assert model_on_device is model  # Should return self

    # Check device - handle JIT models differently
    torch_model = model.model
    if hasattr(torch_model, "device"):
        assert torch_model.device.type == "cpu"

    # Test 4: Encode image and text
    # Create test text inputs
    # Create test images
    mock_image = torch.rand(1, 3, 224, 224)  # Single image tensor
    test_texts = ["This is a test", "Another test text"]

    image_embeddings = model.encode_image(mock_image)
    text_embeddings = model.encode_text(test_texts)

    # Check that the output is a tensor with the expected shape
    assert isinstance(text_embeddings, torch.Tensor)
    assert torch.is_floating_point(text_embeddings)
    assert len(text_embeddings.shape) == 2  # (batch_size, embedding_dim)
    assert text_embeddings.shape[0] == 2  # batch size of 2

    # Test 8: Check compatibility between image and text embeddings
    # The embedding dimensions for image and text should be the same
    assert image_embeddings.shape[1] == text_embeddings.shape[1]

    # Explicitly delete the model to free memory
    del model


# Test function for all multimodal models
@pytest.mark.gpu
@pytest.mark.parametrize("model_class", MULTIMODAL_MODELS)
def test_multimodal_model(model_class):
    """Test a multimodal model.

    This test function runs all tests for a single multimodal model, then releases the model
    to free memory before moving to the next model. This ensures that only one model
    is in memory at a time, which is important for large models that can consume a lot
    of memory.
    """
    # Run all tests for this model
    run_multimodal_model_tests(model_class)
