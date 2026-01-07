from typing import List

import pytest
import torch
from huggingface_hub.errors import GatedRepoError

from lazyslide.models import MODEL_REGISTRY, ModelTask

MODELS = list(MODEL_REGISTRY.keys())

# Dictionary to store model input arguments for FLOPS estimation
MODEL_INPUT_ARGS = {
    "brightness": {"args": [torch.randn(1, 3, 224, 224)], "kwargs": {}},
    "canny": {"args": [torch.randn(1, 3, 224, 224)], "kwargs": {}},
    "cellpose": {"args": [torch.randn(1, 3, 224, 224)], "kwargs": {}},
    "chief": {"args": [torch.randn(1, 3, 224, 224)], "kwargs": {}},
    "chief-slide-encoder": {"args": [torch.randn(100, 768)], "kwargs": {}},
    "conch": {"args": [torch.randn(1, 3, 224, 224)], "kwargs": {}, "method": "encode_image"},
    "conch_v1.5": {"args": [torch.randn(1, 3, 448, 448)], "kwargs": {}, "method": "conch.forward"},
    "contrast": {"args": [torch.randn(1, 3, 224, 224)], "kwargs": {}},
    "ctranspath": {"args": [torch.randn(1, 3, 224, 224)], "kwargs": {}},
    "entropy": {"args": [torch.randn(1, 3, 224, 224)], "kwargs": {}},
    "focus": {"args": [torch.randn(1, 3, 256, 256)], "kwargs": {}},
    "focuslitenn": {"args": [torch.randn(1, 3, 256, 256)], "kwargs": {}},
    "gigapath": {"args": [torch.randn(1, 3, 224, 224)], "kwargs": {}},
    "gigapath-slide-encoder": {"args": [torch.randn(100, 1536), torch.randn(100, 2)], "kwargs": {}},
    "gigatime": {"args": [torch.randn(1, 3, 224, 224)], "kwargs": {}},
    "gpfm": {"args": [torch.randn(1, 3, 224, 224)], "kwargs": {}},
    "grandqc-artifact": {"args": [torch.randn(1, 3, 224, 224)], "kwargs": {}},
    "grandqc-tissue": {"args": [torch.randn(1, 3, 224, 224)], "kwargs": {}},
    "h-optimus-0": {"args": [torch.randn(1, 3, 224, 224)], "kwargs": {}},
    "h-optimus-1": {"args": [torch.randn(1, 3, 224, 224)], "kwargs": {}},
    "h0-mini": {"args": [torch.randn(1, 3, 224, 224)], "kwargs": {}},
    "haralick_texture": {"args": [torch.randn(1, 3, 224, 224)], "kwargs": {}},
    "hest-tissue-segmentation": {"args": [torch.randn(1, 3, 224, 224)], "kwargs": {}},
    "hibou-b": {"args": [], "kwargs": {"pixel_values": torch.randn(1, 3, 224, 224)}},
    "hibou-l": {"args": [], "kwargs": {"pixel_values": torch.randn(1, 3, 224, 224)}},
    "histoplus": {"args": [torch.randn(1, 3, 840, 840)], "kwargs": {}},
    "instanseg": {"args": [torch.randn(1, 3, 224, 224)], "kwargs": {}},
    "madeleine": {"args": [torch.randn(1, 100, 512)], "kwargs": {}},
    "medsiglip": {"args": [], "kwargs": {"pixel_values": torch.randn(1, 3, 448, 448)}, "method": "get_image_features"},
    "midnight": {"args": [torch.randn(1, 3, 224, 224)], "kwargs": {}},
    "musk": {"args": [torch.randn(1, 3, 224, 224)], "kwargs": {}},
    "nulite": {"args": [torch.randn(1, 3, 224, 224)], "kwargs": {}},
    "omiclip": {"args": [torch.randn(1, 3, 224, 224)], "kwargs": {}},
    "path_orchestra": {"args": [torch.randn(1, 3, 224, 224)], "kwargs": {}},
    "pathprofiler": {"args": [torch.randn(1, 3, 512, 512)], "kwargs": {}},
    "pathprofilerqc": {"args": [torch.randn(1, 3, 224, 224)], "kwargs": {}},
    "phikon": {"args": [torch.randn(1, 3, 224, 224)], "kwargs": {}},
    "phikonv2": {"args": [torch.randn(1, 3, 224, 224)], "kwargs": {}},
    "plip": {"args": [], "kwargs": {"pixel_values": torch.randn(1, 3, 224, 224)}, "method": "get_image_features"},
    "prism": {"args": [torch.randn(1, 100, 768)], "kwargs": {}, "method": "model.slide_representations"},
    "rosie": {"args": [torch.randn(1, 3, 224, 224)], "kwargs": {}},
    "sam": {"args": [torch.randn(1, 3, 1024, 1024)], "kwargs": {}},
    "saturation": {"args": [torch.randn(1, 3, 224, 224)], "kwargs": {}},
    "sharpness": {"args": [torch.randn(1, 3, 224, 224)], "kwargs": {}},
    "sobel": {"args": [torch.randn(1, 3, 224, 224)], "kwargs": {}},
    "spider-breast": {"args": [torch.randn(1, 3, 224, 224)], "kwargs": {}},
    "spider-colorectal": {"args": [torch.randn(1, 3, 224, 224)], "kwargs": {}},
    "spider-skin": {"args": [torch.randn(1, 3, 224, 224)], "kwargs": {}},
    "spider-thorax": {"args": [torch.randn(1, 3, 224, 224)], "kwargs": {}},
    "split_rgb": {"args": [torch.randn(1, 3, 224, 224)], "kwargs": {}},
    "titan": {"args": [torch.randn(1, 3, 448, 448)], "kwargs": {}, "method": "conch.forward"},
    "uni": {"args": [torch.randn(1, 3, 224, 224)], "kwargs": {}},
    "uni2": {"args": [torch.randn(1, 3, 224, 224)], "kwargs": {}},
    "virchow": {"args": [torch.randn(1, 3, 224, 224)], "kwargs": {}},
    "virchow2": {"args": [torch.randn(1, 3, 224, 224)], "kwargs": {}},
}


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
        input_config = MODEL_INPUT_ARGS[model_name]
        method = input_config.get("method", "forward")
        _ = model.estimate_flops(input_config["args"], input_config["kwargs"], method=method)

    # Test the to device function
    model_on_device = model.to("cpu")
    assert model_on_device is model  # Should return self
