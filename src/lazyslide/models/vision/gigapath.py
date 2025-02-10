import timm
from huggingface_hub import login
from platformdirs import user_cache_path

from lazyslide.models.base import SlideEncoderModel, TimmModel


class GigaPath(TimmModel):
    name = "GigaPath"

    def __init__(self, model_path=None, token=None):
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
        except ModuleNotFoundError:
            pass

        super().__init__("hf_hub:prov-gigapath/prov-gigapath", token=token)


class GigaPathSlideEncoder(SlideEncoderModel):
    def __init__(self, model_path=None, token=None):
        super().__init__()

        if token is not None:
            login(token)

        from gigapath.slide_encoder import create_model

        model = create_model(
            "hf_hub:prov-gigapath/prov-gigapath",
            "gigapath_slide_enc12l768d",
            1536,
            local_dir=str(user_cache_path("lazyslide")),
        )
        self.model = model

    def encode_slide(self, tile_embed, coordinates):
        return self.model(tile_embed, coordinates).squeeze()
