import math
from pathlib import Path

import torch
import torch.nn as nn


class FocusLiteNN(nn.Module):
    """
    In practice, we found MIN contributes VERY LITTLE to the performance. To achieve
    extreme simplicity, different from Equation 1 in the paper, we only use MAX as the
    nonlinear function. All experimental results in the paper are reported based on the
    model using only MAX.

    The model (class FocusLiteNNMinMax) using weighted MAX and MIN as the nonlinear function
    (Equation 1 in paper) has indistinguishable (slightly better) performance compared to the
    model using only MAX (class FocusLiteNN).
    """

    def __init__(self, num_channel=2):
        super(FocusLiteNN, self).__init__()
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


def load_focuslite_model(device="cpu"):
    model = FocusLiteNN()
    ckpt = torch.load(
        Path(__file__).parent / "focuslitenn-2kernel-mse.pt", map_location=device
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    model = torch.compile(model)
    return model
