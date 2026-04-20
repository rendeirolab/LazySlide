#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "huggingface_hub>=0.21",
#   "torch>=2.7.1",
#   "torchvision>=0.26.0",
# ]
# ///

from pathlib import Path

from export_utils import export_model, verify_exported_dict

workdir = Path(__file__).parent
checkpoint_dir = workdir / "checkpoints"
checkpoint_dir.mkdir(parents=True, exist_ok=True)

export_artifacts = workdir / "export_artifacts"
export_artifacts.mkdir(parents=True, exist_ok=True)

HEST_TISSUE_SEG_EXPORT_PATH = export_artifacts / "hest_tissue_seg_exported.pt2"

# %%
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

weights_file = hf_hub_download(
    repo_id="MahmoodLab/hest-tissue-seg",
    filename="deeplabv3_seg_v4.ckpt",
    repo_type="model",
)

# %%
from torchvision.models.segmentation import deeplabv3_resnet50

model = deeplabv3_resnet50(weights=None)
model.classifier[4] = nn.Conv2d(
    in_channels=256,
    out_channels=2,
    kernel_size=1,
    stride=1,
)

checkpoint = torch.load(weights_file, map_location=torch.device("cpu"))

new_state_dict = {}
for key in checkpoint["state_dict"]:
    if "aux" in key:
        continue
    new_key = key.replace("model.", "")
    new_state_dict[new_key] = checkpoint["state_dict"][key]
model.load_state_dict(new_state_dict)
model.eval()

# %%
# Dynamic batch + spatial dims (H, W); DeepLabV3 is fully convolutional
dynamic_shapes = [
    {0: torch.export.Dim.AUTO, 2: torch.export.Dim.AUTO, 3: torch.export.Dim.AUTO}
]

hest_example_input = torch.randn(2, 3, 128, 128)
export_model(
    model,
    hest_example_input,
    HEST_TISSUE_SEG_EXPORT_PATH,
    dynamic_shapes=dynamic_shapes,
)

# %%
torch.manual_seed(42)
fixed_input = torch.randn(4, 3, 256, 256)

verify_exported_dict(
    model, HEST_TISSUE_SEG_EXPORT_PATH, fixed_input, "HEST tissue segmentation"
)
