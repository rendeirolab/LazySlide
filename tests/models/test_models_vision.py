import numpy as np
import pytest
import torch

from lazyslide.models.base import ImageModel

# Import all model classes
from lazyslide.models.multimodal import CONCH, PLIP, OmiCLIP, Prism, Titan

# Slide encoder models
from lazyslide.models.vision import MadeleineSlideEncoder
from lazyslide.models.vision.chief import CHIEF, CHIEFSlideEncoder
from lazyslide.models.vision.conch import CONCHVision
from lazyslide.models.vision.ctranspath import CTransPath
from lazyslide.models.vision.gigapath import GigaPath
from lazyslide.models.vision.h_optimus import H0Mini, HOptimus0, HOptimus1
from lazyslide.models.vision.hibou import HibouB, HibouL
from lazyslide.models.vision.midnight import Midnight
from lazyslide.models.vision.phikon import Phikon, PhikonV2
from lazyslide.models.vision.plip import PLIPVision
from lazyslide.models.vision.uni import UNI, UNI2
from lazyslide.models.vision.virchow import Virchow, Virchow2

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
VISION_MODELS = [
    skip_if_no_token(Phikon),
    skip_if_no_token(PhikonV2),
    skip_if_no_token(HibouB),
    skip_if_no_token(HibouL),
    skip_if_no_token(CHIEF),
    skip_if_no_token(Midnight),
    skip_if_no_token(CONCH),
    skip_if_no_token(CONCHVision),
    skip_if_no_token(CTransPath),
    skip_if_no_token(PLIP),
    skip_if_no_token(PLIPVision),
    skip_if_no_token(UNI),
    skip_if_no_token(UNI2),
    skip_if_no_token(HOptimus0),
    skip_if_no_token(HOptimus1),
    skip_if_no_token(H0Mini),
    skip_if_no_token(GigaPath),
    skip_if_no_token(Virchow),
    skip_if_no_token(Virchow2),
    skip_if_no_token(OmiCLIP),
    skip_if_no_token(Titan),
]


SLIDE_ENCODER_MODELS = [
    (skip_if_no_token(CHIEFSlideEncoder, CHIEF)),
    (skip_if_no_token(MadeleineSlideEncoder, CONCH)),
    (skip_if_no_token(Prism, Virchow)),
    (skip_if_no_token(Titan, Titan)),
]


def run_vision_model_tests(model_class):
    """Run all tests for a vision model.

    This function runs all tests for a single vision model, then releases the model
    to free memory before returning. This ensures that only one model is in memory
    at a time, which is important for large models that can consume a lot of memory.

    Args:
        model_class: The model class to test
    """
    # Initialize the model
    model = model_class()

    # Test 1: Model initialization
    assert isinstance(model, ImageModel)
    assert hasattr(model, "model")
    assert hasattr(model, "name")

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

    # Create test images
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


# Test function for all vision models
@pytest.mark.gpu
@pytest.mark.parametrize("model_class", VISION_MODELS)
def test_vision_model(model_class):
    """Test a vision model.

    This test function runs all tests for a single vision model, then releases the model
    to free memory before moving to the next model. This ensures that only one model
    is in memory at a time, which is important for large models that can consume a lot
    of memory.
    """
    # Run all tests for this model
    run_vision_model_tests(model_class)


def run_slide_encoder_attrs_tests(slide_encoder, vision_encoder):
    """Run all tests for a slide encoder model.

    This function runs all tests for a single slide encoder model, then releases the model
    to free memory before returning. This ensures that only one model is in memory
    at a time, which is important for large models that can consume a lot of memory.

    Args:
        model_class: The model class to test
    """

    # Test 1: Model initialization
    assert hasattr(slide_encoder, "encode_slide")
    assert hasattr(slide_encoder, "model")
    assert hasattr(slide_encoder, "name")

    # Test 2: To device
    model_on_device = slide_encoder.to("cpu")
    assert model_on_device is slide_encoder  # Should return self

    # Check device - handle JIT models differently
    torch_model = slide_encoder.model
    if hasattr(torch_model, "device"):
        assert torch_model.device.type == "cpu"


def run_slide_encoder_output_tests(slide_encoder, vision_encoder):
    # Create test image
    mock_batch_image = torch.randn(2, 3, 224, 224)  # Example shape [B, C, H, W]
    mock_coords = torch.tensor([[0, 0], [224, 224]]).unsqueeze(
        0
    )  # Example coordinates for two tiles
    base_tile_size = 224  # Example base tile size

    # Get embeddings from the vision encoder
    mock_embeddings = vision_encoder.encode_image(mock_batch_image)

    # Test 4: Encode slide
    output = slide_encoder.encode_slide(
        mock_embeddings.unsqueeze(0),
        coords=mock_coords,
        base_tile_size=base_tile_size,
    )

    # Check that the output is a numpy array with the expected shape
    assert isinstance(output, torch.Tensor)
    assert torch.is_floating_point(output)
    assert len(output.shape) == 2  # (batch_size, embedding_dim)
    assert output.shape[0] == 1  # batch size of 2


def prism_model_output_tests(prism_encoder, virchow_encoder):
    """Run output tests for the Prism model."""
    # Create test image
    mock_batch_image = torch.randn(2, 3, 224, 224)  # Example shape [B, C, H, W]

    # Get embeddings from the vision encoder
    mock_embeddings = virchow_encoder.encode_image(mock_batch_image)

    # Test encode_slide
    output = prism_encoder.encode_slide(
        mock_embeddings.unsqueeze(0),
    )
    assert isinstance(output, dict)
    assert "image_embedding" in output
    assert "image_latents" in output
    image_embedding = output["image_embedding"]
    image_latents = output["image_latents"]
    assert isinstance(image_embedding, torch.Tensor)
    assert isinstance(image_latents, torch.Tensor)
    assert image_embedding.shape[0] == 1
    assert image_latents.shape[0] == 1


@pytest.mark.gpu
@pytest.mark.parametrize(["slide_encoder", "vision_encoder"], SLIDE_ENCODER_MODELS)
def test_slide_encoder_model(slide_encoder, vision_encoder):
    """Test a slide encoder model.

    This test function runs all tests for a single slide encoder model, then releases the model
    to free memory before moving to the next model. This ensures that only one model
    is in memory at a time, which is important for large models that can consume a lot
    of memory.
    """
    # Initialize the slide encoder and vision encoder
    slide_encoder = slide_encoder()
    vision_encoder = vision_encoder()

    run_slide_encoder_attrs_tests(slide_encoder, vision_encoder)
    if slide_encoder.__class__.__name__ == "Prism":
        prism_model_output_tests(slide_encoder, vision_encoder)
    else:
        run_slide_encoder_output_tests(slide_encoder, vision_encoder)

    del slide_encoder
    del vision_encoder
