import torch

from lazyslide.models.base import ModelTask, TimmModel


class Virchow(TimmModel, key="virchow"):
    is_gated = True
    task = ModelTask.vision
    license = "Apache 2.0"
    description = "A foundation model for clinical-grade computational pathology and rare cancers detection"
    commercial = True
    hf_url = "https://huggingface.co/paige-ai/Virchow"
    paper_url = "https://doi.org/10.1038/s41591-024-03141-0"
    bib_key = "Vorontsov2024-di"
    param_size = "631.2M"
    encode_dim = 2560
    _hf_hub_id = "paige-ai/Virchow"

    def __init__(self, model_path=None, token=None):
        from timm.layers import SwiGLUPacked

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


class Virchow2(Virchow, key="virchow2"):
    hf_url = "https://huggingface.co/paige-ai/Virchow2"
    paper_url = "https://doi.org/10.48550/arXiv.2408.00738"
    description = "Scaling self-supervised mixed magnification models in pathology"
    bib_key = "Zimmermann2024-ya"
    license = "CC-BY-NC-ND-4.0"
    commercial = False

    _hf_hub_id = "paige-ai/Virchow2"
