from typing import List

import pytest
import torch
from huggingface_hub.errors import GatedRepoError

from lazyslide.models import MODEL_INPUT_ARGS_CONFIG, MODEL_REGISTRY, ModelTask

MODELS = list(MODEL_REGISTRY.keys())


def _build_model_input_args():
    """Convert shape-based config to tensor-based args for testing."""
    result = {}
    for model_name, config in MODEL_INPUT_ARGS_CONFIG.items():
        test_config = {}
        if "args" in config:
            test_config["args"] = [torch.randn(*shape) for shape in config["args"]]
        if "kwargs" in config:
            test_config["kwargs"] = {k: torch.randn(*v) for k, v in config["kwargs"].items()}
        if "method" in config:
            test_config["method"] = config["method"]
        result[model_name] = test_config
    return result


MODEL_INPUT_ARGS = _build_model_input_args()


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
        # Test estimation of param size
        _ = model.estimate_param_size()
        # Test FLOPS estimation if input arguments are available
        if model_name in MODEL_INPUT_ARGS:
            input_config = MODEL_INPUT_ARGS[model_name]
            method = input_config.get("method", "forward")
            kwargs = input_config.get("kwargs", {})
            _ = model.estimate_flops(method, *input_config["args"], **kwargs)

    # Test the to device function
    model_on_device = model.to("cpu")
    assert model_on_device is model  # Should return self
