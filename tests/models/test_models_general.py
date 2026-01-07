from typing import List

import pytest
import torch
from huggingface_hub.errors import GatedRepoError

from lazyslide.models import MODEL_REGISTRY, ModelTask

MODELS = list(MODEL_REGISTRY.keys())

# Dictionary to store model input arguments for FLOPS estimation
MODEL_INPUT_ARGS = {
    "brightness": {"args": [torch.randn(1, 3, 224, 224)]},
    "canny": {"args": [torch.randn(1, 3, 224, 224)]},
    "cellpose": {"args": [torch.randn(1, 3, 224, 224)]},
    "chief": {"args": [torch.randn(1, 3, 224, 224)]},
    "chief-slide-encoder": {"args": [torch.randn(100, 768)]},
    "conch": {"args": [torch.randn(1, 3, 224, 224)], "method": "encode_image"},
    "conch_v1.5": {"args": [torch.randn(1, 3, 448, 448)], "method": "conch.forward"},
    "contrast": {"args": [torch.randn(1, 3, 224, 224)]},
    "ctranspath": {"args": [torch.randn(1, 3, 224, 224)]},
    "entropy": {"args": [torch.randn(1, 3, 224, 224)]},
    "focus": {"args": [torch.randn(1, 3, 256, 256)]},
    "focuslitenn": {"args": [torch.randn(1, 3, 256, 256)]},
    "gigapath": {"args": [torch.randn(1, 3, 224, 224)]},
    "gigapath-slide-encoder": {"args": [torch.randn(100, 1536), torch.randn(100, 2)]},
    "gigatime": {"args": [torch.randn(1, 3, 224, 224)]},
    "gpfm": {"args": [torch.randn(1, 3, 224, 224)]},
    "grandqc-artifact": {"args": [torch.randn(1, 3, 224, 224)]},
    "grandqc-tissue": {"args": [torch.randn(1, 3, 224, 224)]},
    "h-optimus-0": {"args": [torch.randn(1, 3, 224, 224)]},
    "h-optimus-1": {"args": [torch.randn(1, 3, 224, 224)]},
    "h0-mini": {"args": [torch.randn(1, 3, 224, 224)]},
    "haralick_texture": {"args": [torch.randn(1, 3, 224, 224)]},
    "hest-tissue-segmentation": {"args": [torch.randn(1, 3, 224, 224)]},
    "hibou-b": {"args": [], "kwargs": {"pixel_values": torch.randn(1, 3, 224, 224)}},
    "hibou-l": {"args": [], "kwargs": {"pixel_values": torch.randn(1, 3, 224, 224)}},
    "histoplus": {"args": [torch.randn(1, 3, 840, 840)]},
    "instanseg": {"args": [torch.randn(1, 3, 224, 224)]},
    "madeleine": {"args": [torch.randn(1, 100, 512)]},
    "medsiglip": {"args": [], "kwargs": {"pixel_values": torch.randn(1, 3, 448, 448)}, "method": "get_image_features"},
    "midnight": {"args": [torch.randn(1, 3, 224, 224)]},
    "musk": {"args": [torch.randn(1, 3, 224, 224)]},
    "nulite": {"args": [torch.randn(1, 3, 224, 224)]},
    "omiclip": {"args": [torch.randn(1, 3, 224, 224)]},
    "path_orchestra": {"args": [torch.randn(1, 3, 224, 224)]},
    "pathprofiler": {"args": [torch.randn(1, 3, 512, 512)]},
    "pathprofilerqc": {"args": [torch.randn(1, 3, 224, 224)]},
    "phikon": {"args": [torch.randn(1, 3, 224, 224)]},
    "phikonv2": {"args": [torch.randn(1, 3, 224, 224)]},
    "plip": {"args": [], "kwargs": {"pixel_values": torch.randn(1, 3, 224, 224)}, "method": "get_image_features"},
    "prism": {"args": [torch.randn(1, 100, 768)], "method": "model.slide_representations"},
    "rosie": {"args": [torch.randn(1, 3, 224, 224)]},
    "sam": {"args": [torch.randn(1, 3, 1024, 1024)]},
    "saturation": {"args": [torch.randn(1, 3, 224, 224)]},
    "sharpness": {"args": [torch.randn(1, 3, 224, 224)]},
    "sobel": {"args": [torch.randn(1, 3, 224, 224)]},
    "spider-breast": {"args": [torch.randn(1, 3, 224, 224)]},
    "spider-colorectal": {"args": [torch.randn(1, 3, 224, 224)]},
    "spider-skin": {"args": [torch.randn(1, 3, 224, 224)]},
    "spider-thorax": {"args": [torch.randn(1, 3, 224, 224)]},
    "split_rgb": {"args": [torch.randn(1, 3, 224, 224)]},
    "titan": {"args": [torch.randn(1, 3, 448, 448)], "method": "conch.forward"},
    "uni": {"args": [torch.randn(1, 3, 224, 224)]},
    "uni2": {"args": [torch.randn(1, 3, 224, 224)]},
    "virchow": {"args": [torch.randn(1, 3, 224, 224)]},
    "virchow2": {"args": [torch.randn(1, 3, 224, 224)]},
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
        kwargs = input_config.get("kwargs", {})
        _ = model.estimate_flops(method, *input_config["args"], **kwargs)

    # Test the to device function
    model_on_device = model.to("cpu")
    assert model_on_device is model  # Should return self
