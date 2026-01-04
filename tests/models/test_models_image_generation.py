import pytest
import torch
from huggingface_hub.errors import GatedRepoError

from lazyslide.models import MODEL_REGISTRY, list_models
from lazyslide.models.base import ImageGenerationModel

# List of image generation models to test
IMAGE_GENERATION_MODELS = list_models(task="image_generation")


@pytest.mark.large_runner
@pytest.mark.parametrize("model_name", IMAGE_GENERATION_MODELS)
def test_image_generation_model(model_name):
    # Initialize the model
    try:
        model = MODEL_REGISTRY[model_name]()
    except GatedRepoError:
        pytest.skip(f"{model_name} is not available.")
        return

    # Test 1: Model initialization
    assert isinstance(model, ImageGenerationModel)

    # Test 6: Model prediction
    with torch.inference_mode():
        _ = model.generate()

    # Explicitly delete the model to free memory
    del model
