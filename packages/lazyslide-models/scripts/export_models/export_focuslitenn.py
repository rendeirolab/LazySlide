#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pooch>=1.0",
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

FOCUSLITENN_EXPORT_PATH = export_artifacts / "FocusLiteNN_exported.pt2"

# %%
import pooch

weights_file = pooch.retrieve(
    "https://github.com/icbcbicc/FocusLiteNN/raw/refs/heads/master/pretrained_model/focuslitenn-2kernel-mse.pt",
    path=str(checkpoint_dir),
    known_hash="d28308d1f4859012d7e03209091f1741bcdcb80b7e264c9c6d42ea74d05e35fa",
    fname="focuslitenn-2kernel-mse.pt",
)

# %%
import math

import torch
import torch.nn as nn


class FocusLiteNN(nn.Module):
    """
    A FocusLiteNN model for filtering out-of-focus regions in whole slide images.
    """

    def __init__(self, num_channel=2):
        super().__init__()
        self.num_channel = num_channel
        self.conv = nn.Conv2d(3, self.num_channel, 7, stride=5, padding=1)  # 47x47
        self.maxpool = nn.MaxPool2d(kernel_size=47)
        if self.num_channel > 1:
            self.fc = nn.Conv2d(self.num_channel, 1, 1, stride=1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.conv(x)
        x = -self.maxpool(-x)  # minpooling
        if self.num_channel > 1:
            x = self.fc(x)
        x = x.view(batch_size, -1)
        return x


# %%
ckpt = torch.load(weights_file, map_location="cpu", weights_only=True)
model = FocusLiteNN()
model.load_state_dict(ckpt["state_dict"])
model.eval()


# %%
# Dynamic batch + spatial dims (H, W)
dynamic_shapes = [
    {0: torch.export.Dim.AUTO, 2: torch.export.Dim.AUTO, 3: torch.export.Dim.AUTO}
]

focuslitenn_example_input = torch.randn(2, 3, 256, 256)
export_model(
    model,
    focuslitenn_example_input,
    FOCUSLITENN_EXPORT_PATH,
    dynamic_shapes=dynamic_shapes,
)

# %%
torch.manual_seed(42)
fixed_input = torch.randn(4, 3, 256, 256)

verify_exported(model, FOCUSLITENN_EXPORT_PATH, fixed_input, "FocusLiteNN")
