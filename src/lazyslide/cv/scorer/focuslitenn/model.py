import math
from pathlib import Path

import numpy as np

from lazyslide.cv.scorer.base import ScorerBase, ScoreResult

try:
    import torch

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
except ImportError:

    class FocusLiteNN:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "FocusLiteNN requires torch. You can install it using `pip install torch`."
                "Please restart the kernel after installation."
            )


def load_focuslite_model(device="cpu"):
    model = FocusLiteNN()
    if not hasattr(model, "forward"):
        raise ModuleNotFoundError("To use Focuslite, you need to install pytorch")
    ckpt = torch.load(
        Path(__file__).parent / "focuslitenn-2kernel-mse.pt",
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    # model = torch.compile(model)
    return model


class FocusLite(ScorerBase):
    # The device must be CPU, otherwise this module cannot be serialized
    def __init__(self, threshold=3, device="cpu"):
        from torchvision.transforms import Resize, ToTensor

        # threshold should be between 1 and 12
        if not (1 <= threshold <= 12):
            raise ValueError("threshold should be between 1 and 12")
        self.threshold = threshold
        self.model = load_focuslite_model(device)
        self.to_tensor = ToTensor()
        self.resize = Resize((256, 256), antialias=False)

    def apply(self, patch, mask=None):
        """Higher score means the patch is more clean, range from 0 to 1"""
        arr = self.to_tensor(patch)
        # If the image is not big enough, resize it
        if arr.shape[1] < 256 or arr.shape[2] < 256:
            arr = self.resize(arr)
        arr = torch.stack([arr], dim=0)
        score = self.model(arr)
        score = max(0, np.mean(torch.squeeze(score.cpu().data, dim=1).numpy()))
        return ScoreResult(scores={"focus": score}, qc=score < self.threshold)
