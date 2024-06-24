import math
from pathlib import Path

import numpy as np

from lazyslide.cv.scorer.base import ScorerBase

import lazy_loader

torch = lazy_loader.load("torch")


class FocusLiteNN(torch.nn.Module):
    """
    A FocusLiteNN model for filtering out-of-focus regions in whole slide images.
    """

    def __init__(self, num_channel=2):
        super().__init__()
        self.num_channel = num_channel
        self.conv = torch.nn.Conv2d(
            3, self.num_channel, 7, stride=5, padding=1
        )  # 47x47
        self.maxpool = torch.nn.MaxPool2d(kernel_size=47)
        if self.num_channel > 1:
            self.fc = torch.nn.Conv2d(self.num_channel, 1, 1, stride=1, padding=0)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
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


class FocusLite(ScorerBase):
    name = "focus"

    def __init__(self, threshold=3, device="cpu"):
        from torchvision.transforms import ToTensor

        # threshold should be between 1 and 12
        if not (1 <= threshold <= 12):
            raise ValueError("threshold should be between 1 and 12")
        self.threshold = threshold
        self.model = load_focuslite_model(device)
        self.to_tensor = ToTensor()

    def get_score(self, patch) -> float:
        """Higher score means the patch is more clean, range from 0 to 1"""
        arr = self.to_tensor(patch)
        arr = torch.stack([arr], dim=0)
        score = self.model(arr)
        score = max(0, np.mean(torch.squeeze(score.cpu().data, dim=1).numpy()))
        return score

    def filter(self, scores):
        return scores > self.threshold
