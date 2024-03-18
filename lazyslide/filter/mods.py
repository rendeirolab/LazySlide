import cv2
import torch
import numpy as np
from skimage.measure import blur_effect
from skimage.color import rgb2gray

from .base import FilterBase
from .model import load_focuslite_model


class SobelFilter(FilterBase):
    def __init__(self, grad_mag: float = 20, threshold: float = 0.6):
        self.grad_mag = grad_mag
        self.threshold = threshold

    @staticmethod
    def get_gradient_magnitude(img):
        "Get magnitude of gradient for given image"
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ddepth = cv2.CV_32F
        dx = cv2.Sobel(img, ddepth, 1, 0)
        dy = cv2.Sobel(img, ddepth, 0, 1)
        dxabs = cv2.convertScaleAbs(dx)
        dyabs = cv2.convertScaleAbs(dy)
        mag = cv2.addWeighted(dxabs, 0.5, dyabs, 0.5, 0)
        return mag

    def get_scores(self, patch) -> float:
        """Higher score means the patch is more clean, range from 0 to 1"""
        return np.sum(self.get_gradient_magnitude(patch) >= self.grad_mag) / patch.size

    def filter(self, patch) -> bool:
        return self.get_scores(patch) > self.threshold


class BlurFilter(FilterBase):
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def get_scores(self, patch) -> float:
        """Higher score means the patch is more clean, range from 0 to 1"""
        return blur_effect(rgb2gray(patch))

    def filter(self, patch) -> bool:
        return self.get_scores(patch) < self.threshold


class FocusLiteFilter(FilterBase):
    def __init__(self, threshold: float = 3, device="cpu"):
        from torchvision.transforms import ToTensor

        # threshold should be between 1 and 12
        if not (1 <= threshold <= 12):
            raise ValueError("threshold should be between 1 and 12")
        self.threshold = threshold
        self.model = load_focuslite_model(device)
        self.to_tensor = ToTensor()

    def get_scores(self, patch) -> float:
        """Higher score means the patch is more clean, range from 0 to 1"""
        arr = self.to_tensor(patch)
        arr = torch.stack([arr], dim=0)
        score = self.model(arr)
        score = max(0, np.mean(torch.squeeze(score.cpu().data, dim=1).numpy()))
        return score

    def filter(self, patch) -> bool:
        return self.get_scores(patch) < self.threshold
