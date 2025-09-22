from typing import List

import pytest
from huggingface_hub.errors import GatedRepoError

from lazyslide.models import MODEL_REGISTRY, ModelTask

MODELS = list(MODEL_REGISTRY.keys())


@pytest.mark.large_runner
@pytest.mark.parametrize("model_name", MODELS)
def test_model_init(model_name):
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

    # Test all attributes
    if model.task == ModelTask.cv_feature:
        assert model.name is not None
    else:
        assert model.model is not None
        assert model.name is not None
        assert isinstance(model.task, (ModelTask, List))
        assert model.license is not None
        assert model.commercial is not None

    # Test the to device function
    model_on_device = model.to("cpu")
    assert model_on_device is model  # Should return self

    # Test estimation of param size
    _ = model.estimate_param_size()
