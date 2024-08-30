import torch


class GigaPath(torch.nn.Module):
    def __init__(self, model_path=None, token=None):
        try:
            import timm
            from huggingface_hub import login
        except ImportError:
            raise ImportError(
                "To use gigapath, you need to install timm. You can install it using "
                "`pip install timm."
            )

        super().__init__()

        # Version check
        try:
            from packaging import version

            timm_version = version.parse(timm.__version__)
            minimum_version = version.parse("1.0.3")
            if timm_version < minimum_version:
                raise ImportError(
                    f"Gigapath needs timm >= 1.0.3. You have version {timm_version}."
                    f"Run `pip install --upgrade timm` to install the latest version."
                )
        # If packaging is not installed, skip the version check
        except ImportError:
            pass

        if token is not None:
            login(token)

        model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
        self.model = model

    def forward(self, image):
        return self.model(image)


class GigaPathSlideEncoder(torch.nn.Module):
    def __init__(self, model_path=None, auth_token=None):
        try:
            import timm
            from huggingface_hub import login
            from gigapath.slide_encoder import create_model
        except ImportError:
            raise ImportError(
                "To use GigaPathSlideEncoder, you need to install timm and gigapath. You can install it using "
                "`pip install timm. and pip install git+https://github.com/prov-gigapath/prov-gigapath.git"
            )

        super().__init__()

        if auth_token is not None:
            login(auth_token)

        model = create_model(
            "hf_hub:prov-gigapath/prov-gigapath", "gigapath_slide_enc12l768d", 1536
        )
        self.model = model

    def forward(self, tile_embed, coordinates):
        return self.model(tile_embed, coordinates).squeeze()
