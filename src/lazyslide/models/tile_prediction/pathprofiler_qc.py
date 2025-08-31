import numpy as np
import torch

from ..base import ModelTask, TilePredictionModel


class PathProfilerQC(TilePredictionModel, key="pathprofilerqc"):
    task = ModelTask.tile_prediction
    license = "GPL-3.0"
    description = "Quality assessment of histology images"
    commercial = False
    github_url = "https://github.com/MaryamHaghighat/PathProfiler"
    paper_url = "https://doi.org/10.1038/s41598-022-08351-5"
    bib_key = "Haghighat2022-sy"
    param_size = "11.2M"

    def __init__(self, model_path=None, token=None):
        from huggingface_hub import hf_hub_download

        model_file = hf_hub_download(
            "RendeiroLab/LazySlide-models-gpl",
            "PathProfiler/pathprofiler_patch_quality_jit.pt",
        )

        self.model = torch.jit.load(model_file, map_location="cpu")

    def get_transform(self):
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.v2 import (
            Compose,
            Resize,
            ToDtype,
            ToImage,
        )

        return Compose(
            [
                ToImage(),
                Resize(
                    size=(256, 256),
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                ToDtype(dtype=torch.float32, scale=True),
            ]
        )

    def predict(self, image):
        """
        Predict the class of the input image using the PathProfiler model.
        The model expects a tensor of shape [B, C, H, W].

        | **Suggested Name**       | **Description**                                                    |
        | ------------------------ | ------------------------------------------------------------------ |
        | `diagnostic_quality`     | Whether the image is usable for diagnosis (1 = good quality)       |
        | `visual_cleanliness`     | Whether the image appears normal and free of artefacts (1 = clean) |
        | `focus_issue`            | Degree of focus issue (1 = severe, 0.5 = slight, 0 = none)         |
        | `staining_issue`       | Degree of staining issue (1 = severe, 0.5 = slight, 0 = none)      |
        | `tissue_folding_present` | Whether tissue folding is present (1 = present)                    |
        | `misc_artifacts_present` | Whether other artefacts are present (1 = present)                  |

        """
        names = [
            "diagnostic_quality",
            "visual_cleanliness",
            "focus_issue",
            "staining_issue",
            "tissue_folding_present",
            "misc_artifacts_present",
        ]
        with torch.inference_mode():
            outputs = self.model(image)
            outputs = outputs.T.detach().cpu().numpy()
            return dict(zip(names, outputs))
