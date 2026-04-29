#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pooch>=1.0",
#   "tqdm",
#   "six",
#   "segmentation-models-pytorch==0.3.0",
#   "torch>=2.7.1",
# ]
# ///

from pathlib import Path

from export_utils import export_model, verify_exported

workdir = Path(__file__).parent
checkpoint_dir = workdir / "checkpoints"
checkpoint_dir.mkdir(parents=True, exist_ok=True)

export_artifacts = workdir / "export_artifacts"
export_artifacts.mkdir(parents=True, exist_ok=True)

GRANDQC_MPP1_EXPORT_PATH = export_artifacts / "GrandQC_MPP1_exported.pt2"
GRANDQC_MPP2_EXPORT_PATH = export_artifacts / "GrandQC_MPP2_exported.pt2"
GRANDQC_MPP15_EXPORT_PATH = export_artifacts / "GrandQC_MPP15_exported.pt2"

# %%
import importlib.util
import sys

import pooch
import torch


# Handle timm import compatibility for older saved models
def setup_timm_compatibility():
    """Setup compatibility for timm models saved with older versions."""
    try:
        old_to_new_mappings = {
            "timm.models.layers.activations": "timm.layers.activations",
            "timm.models.layers": "timm.layers",
            "timm.models.efficientnet_blocks": "timm.layers",
            "timm.models.layers.conv_bn_act": "timm.layers.conv_bn_act",
            "timm.models.layers.create_act": "timm.layers.create_act",
            "timm.models.layers.drop": "timm.layers.drop",
            "timm.models.layers.norm": "timm.layers.norm",
            "timm.models.layers.pool2d_same": "timm.layers.pool2d_same",
            "timm.models.layers.squeeze_excite": "timm.layers.squeeze_excite",
        }
        for old_path, new_path in old_to_new_mappings.items():
            if importlib.util.find_spec(old_path) is None:
                try:
                    new_module = importlib.import_module(new_path)
                    sys.modules[old_path] = new_module
                except ImportError:
                    print(f"Warning: Could not map {old_path} to {new_path}")
    except Exception as e:
        print(f"Warning: Could not setup timm compatibility: {e}")


setup_timm_compatibility()

# %%
registry = {
    "GrandQC_MPP1.pth": None,
    "GrandQC_MPP2.pth": None,
    "GrandQC_MPP15.pth": None,
}

ENTRY = pooch.create(
    path=str(checkpoint_dir),
    base_url="https://zenodo.org/records/14041538/files/",
    registry=registry,
)

for key in registry.keys():
    ENTRY.fetch(key, progressbar=True)


# %%
# GrandQC MPP checkpoints are full pickled nn.Module objects (no separate arch definition).
# We load the full model and export it directly.
model_tasks = [
    ("GrandQC_MPP1.pth", GRANDQC_MPP1_EXPORT_PATH, "GrandQC MPP1"),
    ("GrandQC_MPP2.pth", GRANDQC_MPP2_EXPORT_PATH, "GrandQC MPP2"),
    ("GrandQC_MPP15.pth", GRANDQC_MPP15_EXPORT_PATH, "GrandQC MPP15"),
]

# Dynamic batch + spatial dims (H, W); EfficientNet-based models are fully convolutional
dynamic_shapes = [
    {0: torch.export.Dim.AUTO, 2: torch.export.Dim.AUTO, 3: torch.export.Dim.AUTO}
]

example_input = torch.randn(2, 3, 512, 512)

loaded_models = {}
for pth_name, export_path, display_name in model_tasks:
    print(f"\n{'=' * 50}")
    print(f"Processing {display_name}")
    print(f"{'=' * 50}")

    model = torch.load(
        str(checkpoint_dir / pth_name), map_location="cpu", weights_only=False
    )
    model.eval()
    loaded_models[display_name] = model

    export_model(model, example_input, export_path, dynamic_shapes=dynamic_shapes)
    print(f"Exported {display_name} to {export_path}")


# %%
# --- Verification ---
torch.manual_seed(42)
fixed_input = torch.randn(4, 3, 512, 512)

for pth_name, export_path, display_name in model_tasks:
    verify_exported(
        loaded_models[display_name],
        export_path,
        fixed_input,
        display_name,
    )
