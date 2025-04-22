import torch
from timm.layers import SwiGLUPacked

from lazyslide.models.base import TimmModel


class Virchow(TimmModel):
    _hf_hub_id = "paige-ai/Virchow"

    def __init__(self, model_path=None, token=None):
        super().__init__(
            f"hf-hub:{self._hf_hub_id}",
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
            token=token,
        )

    @torch.inference_mode()
    def encode_image(self, img):
        output = self.model(img)
        # CLS token features (1, 768):
        cls_features = output[:, 0]
        # Patch token features (1, 256, 768):
        patch_features = output[:, self.model.num_prefix_tokens :]
        return torch.cat((cls_features, patch_features.mean(1)), dim=-1)


class Virchow2(Virchow):
    _hf_hub_id = "paige-ai/Virchow2"
