import torch

from lazyslide.models.base import ModelTask, TimmModel


class UNI(TimmModel, key="uni"):
    is_gated = True
    task = ModelTask.vision
    license = "CC-BY-NC-ND-4.0"
    description = "General-purpose self-supervised model for pathology"
    commercial = False
    hf_url = "https://huggingface.co/MahmoodLab/UNI"
    github_url = "https://github.com/mahmoodlab/UNI"
    paper_url = "https://doi.org/10.1038/s41591-024-02857-3"
    bib_key = "Chen2024-qt"
    param_size = "303.4M"
    encode_dim = 1024

    def __init__(self, model_path=None, token=None):
        # from huggingface_hub import hf_hub_download
        # model_path = hf_hub_download("MahmoodLab/UNI", filename="pytorch_model.bin")

        if model_path is not None:
            super().__init__(
                "vit_large_patch16_224",
                token=token,
                img_size=224,
                patch_size=16,
                init_values=1e-5,
                num_classes=0,
                dynamic_img_size=True,
                pretrained=False,
            )
            self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        else:
            super().__init__(
                "hf-hub:MahmoodLab/uni",
                token=token,
                init_values=1e-5,
                dynamic_img_size=True,
            )


class UNI2(TimmModel, key="uni2"):
    is_gated = True
    task = ModelTask.vision
    license = "CC-BY-NC-ND-4.0"
    description = "An improved version of UNI"
    commercial = False
    hf_url = "https://huggingface.co/MahmoodLab/UNI2-h"
    github_url = "https://github.com/mahmoodlab/UNI"
    paper_url = "https://doi.org/10.1038/s41591-024-02857-3"
    bib_key = "Chen2024-qt"
    param_size = "681.4M"
    encode_dim = 1536

    def __init__(self, model_path=None, token=None):
        import timm

        timm_kwargs = {
            "img_size": 224,
            "patch_size": 14,
            "depth": 24,
            "num_heads": 24,
            "init_values": 1e-5,
            "embed_dim": 1536,
            "mlp_ratio": 2.66667 * 2,
            "num_classes": 0,
            "no_embed_class": True,
            "mlp_layer": timm.layers.SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
            "reg_tokens": 8,
            "dynamic_img_size": True,
        }

        # from huggingface_hub import hf_hub_download
        # model_path = hf_hub_download("MahmoodLab/UNI2-h", filename="pytorch_model.bin")

        if model_path is not None:
            super().__init__(
                "vit_giant_patch14_224", token=token, pretrained=False, **timm_kwargs
            )
            self.model.load_state_dict(
                torch.load(model_path, map_location="cpu"), strict=True
            )
        else:
            super().__init__("hf-hub:MahmoodLab/UNI2-h", **timm_kwargs)
