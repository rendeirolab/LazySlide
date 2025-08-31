from typing import Literal

import numpy as np
import torch

from ..base import ModelTask, TilePredictionModel

SPIDER_VARIANTS = Literal[
    "breast",
    "colorectal",
    "skin",
    "thorax",
]


class Spider(TilePredictionModel, abstract=True):
    is_gated = True
    task = ModelTask.tile_prediction
    license = "CC BY-NC 4.0"
    commercial = False
    hf_url = "https://huggingface.co/collections/histai/spider-models-and-datasets-6814834eca365b006389c117"
    param_size = "303.9M"

    def __init__(self, variants: SPIDER_VARIANTS, model_path=None, token=None):
        from transformers import AutoModel, AutoProcessor

        self.model = AutoModel.from_pretrained(
            f"histai/SPIDER-{variants}-model", trust_remote_code=True, token=token
        )
        self.processor = AutoProcessor.from_pretrained(
            f"histai/SPIDER-{variants}-model", trust_remote_code=True, token=token
        )

    def predict(self, image):
        """
        Predict the class of the input image using the SPIDER model.
        The model expects a tensor of shape [B, C, H, W].
        """
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.inference_mode():
            outputs = self.model(**inputs)
            predicted_class_names = outputs.predicted_class_names
            prob = outputs.logits.softmax(-1).detach().cpu().numpy()
            return {
                "class": np.asarray(predicted_class_names),
                "prob": prob[:, outputs.label],
            }


class SpiderBreast(Spider, key="spider-breast"):
    description = "Tile classification for breast"

    def __init__(self, model_path=None, token=None):
        super().__init__(variants="breast", model_path=model_path, token=token)


class SpiderColorectal(Spider, key="spider-colorectal"):
    description = "Tile classification for colorectal"

    def __init__(self, model_path=None, token=None):
        super().__init__(variants="colorectal", model_path=model_path, token=token)


class SpiderSkin(Spider, key="spider-skin"):
    description = "Tile classification for skin"

    def __init__(self, model_path=None, token=None):
        super().__init__(variants="skin", model_path=model_path, token=token)


class SpiderThorax(Spider, key="spider-thorax"):
    description = "Tile classification for thorax"

    def __init__(self, model_path=None, token=None):
        super().__init__(variants="thorax", model_path=model_path, token=token)
