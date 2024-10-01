import torch


class UNI(torch.nn.Module):
    def __init__(self, model_path=None, token=None):
        try:
            import timm
        except ImportError:
            raise ImportError(
                "To use UNI, you need to install timm. You can install it using "
                "`pip install timm."
            )

        super().__init__()

        if model_path is not None:
            model = timm.create_model(
                "vit_large_patch16_224",
                img_size=224,
                patch_size=16,
                init_values=1e-5,
                num_classes=0,
                dynamic_img_size=True,
            )
            model.load_state_dict(torch.load(model_path), map_location="cpu")
        else:
            model = timm.create_model("hf-hub:MahmoodLab/uni", hf_auth_token=token)
        self.model = model

    def forward(self, image):
        return self.model(image)
