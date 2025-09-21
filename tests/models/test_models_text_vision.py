import numpy as np
import pytest
import torch
from huggingface_hub.errors import GatedRepoError

# Import multimodal models that implement both encode_image and encode_text
from lazyslide.models import MODEL_REGISTRY, list_models
from lazyslide.models.base import ImageTextModel

# Define lists of models for parametrization
MULTIMODAL_MODELS = list_models(task="multimodal")
MULTIMODAL_MODELS.remove("conch_v1.5")  # Duplicated model name as titan


@pytest.mark.gpu
@pytest.mark.parametrize("model_name", MULTIMODAL_MODELS)
def test_multimodal_model(model_name):
    """Run all tests for a multimodal model.

    This function runs all tests for a single multimodal model, then releases the model
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

    if hasattr(model, "encode_text"):
        # The encode_image has been tested in the test_models_vision.py file
        # Here we only test the encode_text method
        test_texts = ["This is a test", "Another test text"]

        text_embeddings = model.encode_text(test_texts)

        # Check that the output is a tensor with the expected shape
        assert isinstance(text_embeddings, torch.Tensor)
        assert torch.is_floating_point(text_embeddings)
        assert len(text_embeddings.shape) == 2  # (batch_size, embedding_dim)
        assert text_embeddings.shape[0] == 2  # batch size of 2

    # Explicitly delete the model to free memory
    del model
