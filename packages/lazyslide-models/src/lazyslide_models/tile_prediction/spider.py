from typing import Literal

import numpy as np
import torch

from .._model_registry import register
from ..base import ModelTask, TilePredictionModel

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
        self.model.eval()
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
            return {"class": np.asarray(predicted_class_names), "prob": prob.max(1)}


shared_info = dict(
    is_gated=True,
    task=ModelTask.tile_prediction,
    license="CC BY-NC 4.0",
    commercial=False,
    hf_url="https://huggingface.co/collections/histai/spider-models-and-datasets-6814834eca365b006389c117",
    param_size="303.9M",
)


@register(
    key="spider-breast",
    description="Tile classification for breast",
    flops="164.85G",
    **shared_info,
)
class SpiderBreast(Spider):
    """
    The output classes are:

    - Adenosis
    - Benign phyllodes tumor
    - Ductal carcinoma in situ (high-grade)
    - Ductal carcinoma in situ (low-grade)
    - Fat
    - Fibroadenoma
    - Fibrocystic changes
    - Fibrosis
    - Invasive non-special type carcinoma
    - Lipogranuloma
    - Lobular invasive carcinoma
    - Malignant phyllodes tumor
    - Necrosis
    - Normal ducts
    - Normal lobules
    - Sclerosing adenosis
    - Typical ductal hyperplasia
    - Vessels

    """

    def __init__(self, model_path=None, token=None):
        super().__init__(variants="breast", model_path=model_path, token=token)


@register(
    key="spider-colorectal",
    description="Tile classification for colorectal",
    flops="164.85G",
    **shared_info,
)
class SpiderColorectal(Spider):
    """
    The output classes are:

    - Adenocarcinoma high grade
    - Adenocarcinoma low grade
    - Adenoma high grade
    - Adenoma low grade
    - Fat
    - Hyperplastic polyp
    - Inflammation
    - Mucus
    - Muscle
    - Necrosis
    - Sessile serrated lesion
    - Stroma healthy
    - Vessels

    """

    def __init__(self, model_path=None, token=None):
        super().__init__(variants="colorectal", model_path=model_path, token=token)


@register(
    key="spider-skin",
    description="Tile classification for skin",
    flops="164.85G",
    **shared_info,
)
class SpiderSkin(Spider):
    """
    The output classes are:

    - Actinic keratosis
    - Apocrine glands
    - Basal cell carcinoma
    - Carcinoma in situ
    - Collagen
    - Epidermis
    - Fat
    - Follicle
    - Inflammation
    - Invasive melanoma
    - Kaposi’s sarcoma
    - Keratin
    - Melanoma in situ
    - Mercel cell carcinoma
    - Muscle
    - Necrosis
    - Nerves
    - Nevus
    - Sebaceous gland
    - Seborrheic keratosis
    - Solar elastosis
    - Squamous cell carcinoma
    - Vessels
    - Wart

    """

    def __init__(self, model_path=None, token=None):
        super().__init__(variants="skin", model_path=model_path, token=token)


@register(
    key="spider-thorax",
    description="Tile classification for thorax",
    flops="164.85G",
    **shared_info,
)
class SpiderThorax(Spider):
    """
    The output classes are:

    - Alveoli
    - Bronchial cartilage
    - Bronchial glands
    - Chronic inflammation + fibrosis
    - Detritus
    - Fibrosis
    - Hemorrhage
    - Lymph node
    - Pigment
    - Pleura
    - Tumor non-small cell
    - Tumor small cell
    - Tumor soft
    - Vessel

    """

    def __init__(self, model_path=None, token=None):
        super().__init__(variants="thorax", model_path=model_path, token=token)
