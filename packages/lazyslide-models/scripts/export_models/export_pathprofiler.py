#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "gdown==6.0.0",
#   "torch>=2.7.1",
#   "torchvision>=0.26.0",
# ]
# ///

from pathlib import Path

from export_utils import export_model, verify_exported

workdir = Path(__file__).parent
checkpoint_dir = workdir / "checkpoints"
checkpoint_dir.mkdir(parents=True, exist_ok=True)

export_artifacts = workdir / "export_artifacts"
export_artifacts.mkdir(parents=True, exist_ok=True)

TISSUE_SEG_CHECKPOINT = checkpoint_dir / "checkpoint_ts.pth"
PATCH_QUALITY_CHECKPOINT = checkpoint_dir / "checkpoint_106.pth"

TISSUE_SEG_EXPORT_PATH = export_artifacts / "pathprofiler_tissue_seg_exported.pt2"
PATCH_QUALITY_EXPORT_PATH = export_artifacts / "pathprofiler_patch_quality_exported.pt2"

# %%
import gdown

PATHPROFILER_QC_CKP_URL = (
    "https://drive.google.com/file/d/13egPkDufR6W4aTBUAAf8uV6zQxwdBx6r/view?usp=sharing"
)

PATHPROFILER_TISSUE_SEG_CKP_URL = (
    "https://drive.google.com/file/d/1otWor5WnaJ4W9ynTOF1XS755CsxEa4qj/view?usp=sharing"
)

if not TISSUE_SEG_CHECKPOINT.exists():
    gdown.download(
        url=PATHPROFILER_TISSUE_SEG_CKP_URL,
        output=str(TISSUE_SEG_CHECKPOINT),
        quiet=False,
    )

if not PATCH_QUALITY_CHECKPOINT.exists():
    gdown.download(
        url=PATHPROFILER_QC_CKP_URL, output=str(PATCH_QUALITY_CHECKPOINT), quiet=False
    )

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class UNet_down_block(nn.Module):
    def __init__(self, input_channel, output_channel, down_size):
        super(UNet_down_block, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, output_channel, 3, padding=1)
        self.bn1 = nn.InstanceNorm2d(output_channel)
        self.conv2 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn2 = nn.InstanceNorm2d(output_channel)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.down_size = down_size

    def forward(self, x):
        if self.down_size:
            x = self.max_pool(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class UNet_up_block(nn.Module):
    def __init__(self, prev_channel, input_channel, output_channel):
        super(UNet_up_block, self).__init__()
        self.up_sampling = nn.Upsample(scale_factor=2, mode="bilinear")
        self.conv1 = nn.Conv2d(
            prev_channel + input_channel, output_channel, 3, padding=1
        )
        self.bn1 = nn.InstanceNorm2d(output_channel)
        self.conv2 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn2 = nn.InstanceNorm2d(output_channel)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(p=0.2)

    def forward(self, prev_feature_map, x):
        x = self.up_sampling(x)
        x = torch.cat((x, self.dropout(prev_feature_map)), dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.down_block1 = UNet_down_block(3, 16, False)
        self.down_block2 = UNet_down_block(16, 32, True)
        self.down_block3 = UNet_down_block(32, 64, True)
        self.down_block4 = UNet_down_block(64, 128, True)
        self.down_block5 = UNet_down_block(128, 256, True)
        self.down_block6 = UNet_down_block(256, 512, True)
        self.down_block7 = UNet_down_block(512, 1024, True)

        self.mid_conv1 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.bn1 = nn.InstanceNorm2d(1024)
        self.mid_conv2 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.bn2 = nn.InstanceNorm2d(1024)

        self.up_block1 = UNet_up_block(512, 1024, 512)
        self.up_block2 = UNet_up_block(256, 512, 256)
        self.up_block3 = UNet_up_block(128, 256, 128)
        self.up_block4 = UNet_up_block(64, 128, 64)
        self.up_block5 = UNet_up_block(32, 64, 32)
        self.up_block6 = UNet_up_block(16, 32, 16)

        self.last_conv1 = nn.Conv2d(16, 16, 3, padding=1)
        self.last_bn = nn.InstanceNorm2d(16)
        self.last_conv2 = nn.Conv2d(16, 2, 1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.down_block1(x)
        x2 = self.down_block2(x1)
        x3 = self.down_block3(x2)
        x4 = self.down_block4(x3)
        x5 = self.down_block5(x4)
        x6 = self.down_block6(x5)
        x7 = self.down_block7(x6)
        x7 = self.relu(self.bn1(self.mid_conv1(x7)))
        x7 = self.relu(self.bn2(self.mid_conv2(x7)))
        x = self.up_block1(x6, x7)
        x = self.up_block2(x5, x)
        x = self.up_block3(x4, x)
        x = self.up_block4(x3, x)
        x = self.up_block5(x2, x)
        x = self.up_block6(x1, x)
        x = self.relu(self.last_bn(self.last_conv1(x)))
        x = self.last_conv2(x)
        return x


class ResNet18(nn.Module):
    def __init__(self, n_classes):
        super(ResNet18, self).__init__()
        resnet = models.resnet18()
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.down = nn.Linear(512, n_classes)

    def forward(self, x):
        x = self.resnet(x)
        x = F.relu(x.view(x.size(0), x.size(1)))
        x = self.down(x)
        return x


# %%
# --- Tissue segmentation model ---
tissue_seg_net = UNet()
tissue_seg_net = nn.DataParallel(tissue_seg_net)
state_dict = torch.load(TISSUE_SEG_CHECKPOINT, map_location="cpu")
tissue_seg_net.load_state_dict(state_dict["state_dict"])
tissue_seg_model = tissue_seg_net.module  # unwrap DataParallel

# UNet is fully convolutional — dynamic batch + spatial dims (H, W)
image_dynamic_shapes = [
    {0: torch.export.Dim.AUTO, 2: torch.export.Dim.AUTO, 3: torch.export.Dim.AUTO}
]

tissue_seg_example_input = torch.randn(2, 3, 512, 512)
export_model(
    tissue_seg_model,
    tissue_seg_example_input,
    TISSUE_SEG_EXPORT_PATH,
    dynamic_shapes=image_dynamic_shapes,
)

# %%
# --- Patch quality classification model ---
patch_quality_model = ResNet18(n_classes=6)
checkpoint = torch.load(PATCH_QUALITY_CHECKPOINT, map_location=torch.device("cpu"))
patch_quality_model.load_state_dict(checkpoint["state_dict"])

# ResNet18 ends with global avg pool — dynamic batch only (spatial fixed at 224×224)
patch_dynamic_shapes = [{0: torch.export.Dim.AUTO}]

patch_quality_example_input = torch.randn(2, 3, 224, 224)
export_model(
    patch_quality_model,
    patch_quality_example_input,
    PATCH_QUALITY_EXPORT_PATH,
    dynamic_shapes=patch_dynamic_shapes,
)

# %%
# --- Verification ---
torch.manual_seed(42)
fixed_tissue_input = torch.randn(4, 3, 512, 512)
fixed_patch_input = torch.randn(4, 3, 224, 224)

# Reload tissue seg model for verification
tissue_seg_verify = UNet()
tissue_seg_verify = nn.DataParallel(tissue_seg_verify)
state_dict = torch.load(TISSUE_SEG_CHECKPOINT, map_location="cpu")
tissue_seg_verify.load_state_dict(state_dict["state_dict"])
tissue_seg_verify = tissue_seg_verify.module

verify_exported(
    tissue_seg_verify,
    TISSUE_SEG_EXPORT_PATH,
    fixed_tissue_input,
    "PathProfiler tissue segmentation",
)

# Reload patch quality model for verification
patch_quality_verify = ResNet18(n_classes=6)
checkpoint = torch.load(PATCH_QUALITY_CHECKPOINT, map_location=torch.device("cpu"))
patch_quality_verify.load_state_dict(checkpoint["state_dict"])

verify_exported(
    patch_quality_verify,
    PATCH_QUALITY_EXPORT_PATH,
    fixed_patch_input,
    "PathProfiler patch quality",
)
