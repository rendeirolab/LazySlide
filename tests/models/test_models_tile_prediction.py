import numpy as np
import pytest
import torch
from huggingface_hub.errors import GatedRepoError

from lazyslide.models import MODEL_REGISTRY, list_models
from lazyslide.models.base import TilePredictionModel

# Define lists of models for parametrization
CV_FEATURE_MODELS = list_models(task="cv_feature")

TILE_PRED_MODELS = list_models(task="tile_prediction")


@pytest.mark.large_runner
@pytest.mark.parametrize("model_name", CV_FEATURE_MODELS)
def test_cv_feature_model(model_name):
    """Run all tests for a CV feature model.

    This function runs all tests for a single CV feature model, then releases the model
    to free memory before returning.

    Args:
        model_class: The model class to test
    """
    # Initialize the model
    # Initialize the model
    try:
        model = MODEL_REGISTRY[model_name]()
    except GatedRepoError:
        pytest.skip(f"{model_name} is not available.")
        return

    # Test 1: Model initialization
    assert isinstance(model, TilePredictionModel)
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


@pytest.mark.large_runner
@pytest.mark.parametrize("model_name", TILE_PRED_MODELS)
def test_tile_prediction_model(model_name):
    """Run all tests for a Hugging Face model.

    This function runs all tests for a single Hugging Face model, then releases the model
    to free memory before returning.

    Args:
        model_class: The model class to test
    """
    # Initialize the model
    # Initialize the model
    try:
        model = MODEL_REGISTRY[model_name]()
    except GatedRepoError:
        pytest.skip(f"{model_name} is not available.")
        return

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
        if model_name == "focus":
            assert key == "focus"
            assert value.shape == (1,)
        elif model_name == "PathProfilerQC":
            assert key in [
                "diagnostic_quality",
                "visual_cleanliness",
                "focus_issue",
                "staining_issue",
                "tissue_folding_present",
                "misc_artifacts_present",
            ]
            assert value.shape == (1,)
        elif model_name in [
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
