import numpy as np
import torch

from ..base import TilePredictionModel


class FocusLiteNN(TilePredictionModel):
    def __init__(self, model_path=None, token=None):
        from huggingface_hub import hf_hub_download

        model_file = hf_hub_download(
            "RendeiroLab/LazySlide-models", "FocusLiteNN/focuslitenn_jit.pt"
        )
        self.model = torch.jit.load(model_file, map_location="cpu")
        self.model.eval()

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
        Predict the focus score for the input image using the FocusLiteNN model.
        The model expects a tensor of shape [B, C, H, W].
        """
        with torch.inference_mode():
            output = self.model(image)
            # Clip the output to > 0
            output = torch.clamp(output, min=0)
            return {"focus": np.asarray(output.squeeze(-1))}
