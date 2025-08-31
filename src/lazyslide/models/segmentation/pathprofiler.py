import cv2
import numpy as np
import torch

from ..base import ModelTask, SegmentationModel


class CLAHE:
    # histogram equalisation
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __call__(self, img):
        img = np.asarray(img)
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        HSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        HSV[:, :, 0] = self.clahe.apply(HSV[:, :, 0])
        img = cv2.cvtColor(HSV, cv2.COLOR_HSV2RGB)

        if img.shape[-1] == 3:  # If the image is grayscale
            img = img.transpose(2, 0, 1)  # Convert to C, H, W format
        return torch.tensor(img)


class PathProfilerTissueSegmentation(SegmentationModel, key="pathprofiler"):
    """
    Tissue segmentation model from PathProfiler.
    This model works at mpp=2.5 or 1.25
    """

    task = ModelTask.segmentation
    license = "GPL-3.0"
    description = "Tissue segmentation model from PathProfiler"
    commercial = False
    github_url = "https://github.com/MaryamHaghighat/PathProfiler"
    paper_url = "https://doi.org/10.1038/s41598-022-08351-5"
    bib_key = "Haghighat2022-sy"
    param_size = "50.3M"

    def __init__(self, model_path=None, token=None):
        from huggingface_hub import hf_hub_download

        model_file = hf_hub_download(
            "RendeiroLab/LazySlide-models-gpl",
            "PathProfiler/pathprofiler_tissue_seg_jit.pt",
        )

        self.model = torch.jit.load(model_file, map_location="cpu")

    def get_transform(self):
        from torchvision.transforms.v2 import (
            Compose,
            Normalize,
            ToDtype,
            ToImage,
        )

        return Compose(
            [
                CLAHE(),
                ToImage(),
                ToDtype(dtype=torch.float32, scale=True),
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def segment(self, image):
        with torch.inference_mode():
            return {"probability_map": self.model(image).softmax(1)}

    def supported_output(self):
        return ("probability_map",)
