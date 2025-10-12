from typing import Literal

import numpy as np
import torch

from ...base import ModelTask, SegmentationModel
from .postprocess import np_hv_postprocess


class NuLite(SegmentationModel, key="nulite"):
    task = ModelTask.segmentation
    license = ["Apache 2.0", "CC-BY-NC-SA-4.0"]
    description = "Nuclei instance segmentation and classification"
    commercial = False
    github_url = "https://github.com/CosmoIknosLab/NuLite"
    paper_url = "https://doi.org/10.48550/arXiv.2408.01797"
    bib_key = "Tommasino2024-tg"
    param_size = "47.9M"

    def __init__(
        self,
        variant: Literal["H", "M", "T"] = "H",
    ):
        from huggingface_hub import hf_hub_download

        model_file = hf_hub_download(
            "RendeiroLab/LazySlide-models", f"nulite/NuLite_{variant}_jit.pt"
        )

        self.model = torch.jit.load(model_file, map_location="cpu")
        self.model.eval()

    def get_transform(self):
        from torchvision.transforms.v2 import Compose, Normalize, ToDtype, ToImage

        return Compose(
            [
                ToImage(),
                ToDtype(dtype=torch.float32, scale=True),
                Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

    # @torch.inference_mode()
    def segment(self, image):
        with torch.inference_mode():
            output = self.model(image)
        # return output
        # postprocess the output
        flattened = [
            dict(zip(output.keys(), values)) for values in zip(*output.values())
        ]

        instances_maps = []
        prob_maps = []
        for batch in flattened:
            # instance_map = nulite_preprocess(batch)  # Numpy array
            instance_map = np_hv_postprocess(
                batch["nuclei_binary_map"].softmax(0).detach().cpu().numpy()[1],
                batch["hv_map"].detach().cpu().numpy(),
            )
            prob_map = (
                batch["nuclei_type_map"].softmax(0).detach().cpu().numpy()
            )  # Skip background
            instances_maps.append(instance_map)
            prob_maps.append(prob_map)

        return {
            "instance_map": np.array(instances_maps),
            "class_map": np.array(prob_maps),
        }

    def supported_output(self):
        return ["instance_map", "class_map"]

    def get_classes(self):
        return {
            0: "Background",
            1: "Neoplastic",
            2: "Inflammatory",
            3: "Connective",
            4: "Dead",
            5: "Epithelial",
        }
