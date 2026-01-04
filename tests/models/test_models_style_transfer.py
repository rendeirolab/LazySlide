import numpy as np
import pytest
import torch
from huggingface_hub.errors import GatedRepoError

from lazyslide.models import MODEL_REGISTRY, list_models
from lazyslide.models.base import StyleTransferModel

# List of style transfer models to test
STYLE_TRANSFER_MODELS = list_models(task="style_transfer")


@pytest.mark.large_runner
@pytest.mark.parametrize("model_name", STYLE_TRANSFER_MODELS)
def test_style_transfer_model(model_name):
    # Initialize the model
    try:
        model = MODEL_REGISTRY[model_name]()
    except GatedRepoError:
        pytest.skip(f"{model_name} is not available.")
        return

    # Test 1: Model initialization
    assert isinstance(model, StyleTransferModel)

    # Test 3: Get transform
    transform = model.get_transform()
    mock_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    mock_batch = torch.rand(2, 3, 224, 224)

    if transform is not None:
        image_tensor = transform(mock_image).unsqueeze(0)  # Add batch dimension
        _ = transform(mock_batch)
    else:
        # If no transform, create a tensor directly
        image_tensor = torch.rand(1, 3, 224, 224)
        _ = torch.rand(2, 3, 224, 224)

    # Test 6: Model prediction
    with torch.inference_mode():
        output = model.predict(image_tensor)

    # Check output shape and type
    assert isinstance(output, torch.Tensor)
    assert output.shape[0] == 1  # batch size
    # assert output.shape[1] == 50  # ROSIE outputs 50 channels
    assert torch.is_floating_point(output)

    # Explicitly delete the model to free memory
    del model
