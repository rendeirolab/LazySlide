import numpy as np
import pytest
import torch
from huggingface_hub.errors import GatedRepoError

from lazyslide.models import MODEL_REGISTRY, ImageModel, list_models

# Define lists of models for parametrization
VISION_MODELS = list_models(task="vision")

SLIDE_ENCODER_MODELS = list_models(task="slide_encoder")


@pytest.mark.gpu
@pytest.mark.parametrize("model_name", VISION_MODELS)
def test_vision_encoder_model(model_name):
    """Run all tests for a vision encoder model."""
    # Initialize the model
    try:
        model = MODEL_REGISTRY[model_name]()
    except GatedRepoError:
        pytest.skip(f"{model_name} is not available.")
        return
    except ModuleNotFoundError:
        pytest.skip(f"{model_name} has dependencies that are not installed.")
        return
    except NotImplementedError:
        pytest.skip(f"{model_name} maybe deprecated or not supported yet.")
        return

    assert isinstance(model, ImageModel)

    transform = model.get_transform()
    mock_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    mock_batch = torch.rand(2, 3, 224, 224)

    # Test 4: Encode image
    # Apply transform if available
    if transform is not None:
        image_tensor = transform(mock_image).unsqueeze(0)  # Add batch dimension
        batch_tensor = transform(mock_batch)
    else:
        # If no transform, create a tensor directly
        image_tensor = torch.rand(1, 3, 224, 224)
        batch_tensor = torch.rand(2, 3, 224, 224)

    # Test encode_image
    embedding = model.encode_image(image_tensor)

    # Check that the output is a numpy array with the expected shape
    assert isinstance(embedding, torch.Tensor)
    assert torch.is_floating_point(embedding)
    assert len(embedding.shape) == 2  # (batch_size, embedding_dim)
    assert embedding.shape[0] == 1  # batch size of 1

    # Test 5: Encode image batch
    # Test encode_image with a batch
    embedding = model.encode_image(batch_tensor)

    # Check that the output is a numpy array with the expected shape
    assert isinstance(embedding, torch.Tensor)
    assert torch.is_floating_point(embedding)
    assert len(embedding.shape) == 2  # (batch_size, embedding_dim)
    assert embedding.shape[0] == 2  # batch size of 2

    # Explicitly delete the model to free memory
    del model


@pytest.mark.gpu
@pytest.mark.parametrize("slide_encoder", SLIDE_ENCODER_MODELS)
def test_slide_encoder_model(slide_encoder):
    """Test a slide encoder model.

    This test function runs all tests for a single slide encoder model, then releases the model
    to free memory before moving to the next model. This ensures that only one model
    is in memory at a time, which is important for large models that can consume a lot
    of memory.
    """
    # Initialize the slide encoder and vision encoder
    try:
        slide_encoder = MODEL_REGISTRY[slide_encoder]()
    except GatedRepoError:
        pytest.skip(f"{slide_encoder} is not available.")
        return
    except ModuleNotFoundError:
        pytest.skip(f"{slide_encoder} has dependencies that are not installed.")
        return
    except NotImplementedError:
        pytest.skip(f"{slide_encoder} maybe deprecated or not supported yet.")
        return
    assert hasattr(slide_encoder, "vision_encoder")
    vision_encoder = MODEL_REGISTRY[slide_encoder.vision_encoder]()

    mock_batch_image = torch.randn(2, 3, 224, 224)  # Example shape [B, C, H, W]
    mock_coords = torch.tensor([[0, 0], [224, 224]]).unsqueeze(
        0
    )  # Example coordinates for two tiles
    base_tile_size = 224  # Example base tile size

    # Get embeddings from the vision encoder
    mock_embeddings = vision_encoder.encode_image(mock_batch_image)

    # Test 4: Encode slide
    _ = slide_encoder.encode_slide(
        mock_embeddings.unsqueeze(0),
        coords=mock_coords,
        base_tile_size=base_tile_size,
    )

    del slide_encoder
    del vision_encoder
