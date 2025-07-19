import numpy as np
import pytest
import torch

from lazyslide.models.base import TilePredictionModel

# Import all tile prediction models
from lazyslide.models.tile_prediction.cv_features import (
    Brightness,
    Canny,
    Contrast,
    Entropy,
    HaralickTexture,
    Saturation,
    Sharpness,
    Sobel,
    SplitRGB,
)
from lazyslide.models.tile_prediction.focuslitenn import FocusLiteNN
from lazyslide.models.tile_prediction.pathprofiler_qc import PathProfilerQC
from lazyslide.models.tile_prediction.spider import (
    SpiderBreast,
    SpiderColorectal,
    SpiderSkin,
    SpiderThorax,
)

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
CV_FEATURE_MODELS = [
    pytest.param(Brightness),
    pytest.param(Contrast),
    pytest.param(Sharpness),
    pytest.param(Sobel),
    pytest.param(Canny),
    pytest.param(Entropy),
    pytest.param(Saturation),
    pytest.param(SplitRGB),
    pytest.param(HaralickTexture),
]

HF_MODELS = [
    skip_if_no_token(FocusLiteNN),
    skip_if_no_token(PathProfilerQC),
    skip_if_no_token(SpiderBreast),
    skip_if_no_token(SpiderColorectal),
    skip_if_no_token(SpiderSkin),
    skip_if_no_token(SpiderThorax),
]


def run_cv_feature_model_tests(model_class):
    """Run all tests for a CV feature model.

    This function runs all tests for a single CV feature model, then releases the model
    to free memory before returning.

    Args:
        model_class: The model class to test
    """
    # Initialize the model
    model = model_class()

    # Test 1: Model initialization
    assert isinstance(model, TilePredictionModel)
    assert hasattr(model, "key") or model.__class__.__name__ == "SplitRGB"

    # Create test images
    mock_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    mock_batch = np.random.randint(0, 255, (2, 224, 224, 3)).astype(np.uint8)

    # Test 2: Predict with single image
    result = model.predict(mock_image)

    # Check that the output is a dictionary of numpy arrays
    assert isinstance(result, dict)
    for key, value in result.items():
        assert isinstance(value, np.ndarray)
        assert value.shape == (1,) or value.ndim == 1

    # Test 3: Predict with batch of images
    result = model.predict(mock_batch)

    # Check that the output is a dictionary of numpy arrays
    assert isinstance(result, dict)
    for key, value in result.items():
        assert isinstance(value, np.ndarray)
        assert value.shape == (2,) or value.shape[0] == 2

    # Explicitly delete the model to free memory
    del model


def run_hf_model_tests(model_class):
    """Run all tests for a Hugging Face model.

    This function runs all tests for a single Hugging Face model, then releases the model
    to free memory before returning.

    Args:
        model_class: The model class to test
    """
    # Initialize the model
    model = model_class()

    # Test 1: Model initialization
    assert isinstance(model, TilePredictionModel)
    assert hasattr(model, "model")

    # Test 2: Get transform
    transform = model.get_transform()
    if transform is not None:
        assert callable(transform)

    # Create test images
    mock_image = np.random.randn(224, 224, 3)

    # Apply transform
    if transform is not None:
        image_tensor = transform(mock_image).unsqueeze(0)  # Add batch dimension
    else:
        image_tensor = torch.randn(1, 3, 224, 224)

    # Test 3: Predict
    result = model.predict(image_tensor)

    # Check that the output is a dictionary of numpy arrays
    assert isinstance(result, dict)
    for key, value in result.items():
        assert isinstance(value, np.ndarray)

        # Check shape based on model type
        if model_class.__name__ == "FocusLiteNN":
            assert key == "focus"
            assert value.shape == (1,)
        elif model_class.__name__ == "PathProfilerQC":
            assert key in [
                "diagnostic_quality",
                "visual_cleanliness",
                "focus_issue",
                "staining_issue",
                "tissue_folding_present",
                "misc_artifacts_present",
            ]
            assert value.shape == (1,)
        elif model_class.__name__ in [
            "SpiderBreast",
            "SpiderColorectal",
            "SpiderSkin",
            "SpiderThorax",
        ]:
            assert key in ["class", "prob"]
            if key == "class":
                assert isinstance(value[0], str) or isinstance(value[0], np.str_)
            elif key == "prob":
                assert value.shape == (1,)

    # Explicitly delete the model to free memory
    del model


# Test function for CV feature models
@pytest.mark.gpu
@pytest.mark.parametrize("model_class", CV_FEATURE_MODELS)
def test_cv_feature_model(model_class):
    """Test a CV feature model."""
    run_cv_feature_model_tests(model_class)


# Test function for Hugging Face models
@pytest.mark.gpu
@pytest.mark.parametrize("model_class", HF_MODELS)
def test_hf_model(model_class):
    """Test a Hugging Face model."""
    run_hf_model_tests(model_class)
