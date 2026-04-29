#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pooch>=1.0",
#   "segmentation-models-pytorch==0.3.0",
#   "six",
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

GRANDQC_TISSUE_SEG_EXPORT_PATH = export_artifacts / "GrandQC_tissue_seg_exported.pt2"

# %%
import pooch
import segmentation_models_pytorch as smp
import torch

# Download weights
weights_file = pooch.retrieve(
    "https://zenodo.org/records/14507273/files/Tissue_Detection_MPP10.pth",
    path=str(checkpoint_dir),
    known_hash=None,
    fname="Tissue_Detection_MPP10.pth",
)

# %%
# Create and load the base model
model = smp.create_model(
    arch="unetplusplus",
    encoder_name="timm-efficientnet-b0",
    encoder_weights="imagenet",
    in_channels=3,
    classes=2,
    activation=None,
)

model.load_state_dict(
    torch.load(weights_file, map_location=torch.device("cpu"), weights_only=True)
)
model.eval()

# %%
# Dynamic batch + spatial dims (H, W); UNet++ is fully convolutional
dynamic_shapes = [
    {0: torch.export.Dim.AUTO, 2: torch.export.Dim.AUTO, 3: torch.export.Dim.AUTO}
]

grandqc_tissue_example_input = torch.randn(2, 3, 224, 224)
export_model(
    model,
    grandqc_tissue_example_input,
    GRANDQC_TISSUE_SEG_EXPORT_PATH,
    dynamic_shapes=dynamic_shapes,
)

# %%
torch.manual_seed(42)
fixed_input = torch.randn(4, 3, 224, 224)

verify_exported(
    model,
    GRANDQC_TISSUE_SEG_EXPORT_PATH,
    fixed_input,
    "GrandQC tissue segmentation",
)
