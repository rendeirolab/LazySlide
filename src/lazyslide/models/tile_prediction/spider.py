from typing import Literal

import numpy as np
import torch

from ..base import TilePredictionModel

SPIDER_VARIANTS = Literal[
    "breast",
    "colorectal",
    "skin",
    "thorax",
]


class Spider(TilePredictionModel):
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


class SpiderBreast(Spider):
    def __init__(self, model_path=None, token=None):
        super().__init__(variants="breast", model_path=model_path, token=token)


class SpiderColorectal(Spider):
    def __init__(self, model_path=None, token=None):
        super().__init__(variants="colorectal", model_path=model_path, token=token)


class SpiderSkin(Spider):
    def __init__(self, model_path=None, token=None):
        super().__init__(variants="skin", model_path=model_path, token=token)


class SpiderThorax(Spider):
    def __init__(self, model_path=None, token=None):
        super().__init__(variants="thorax", model_path=model_path, token=token)
