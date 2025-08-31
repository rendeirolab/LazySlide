from platformdirs import user_cache_path

from lazyslide.models.base import ModelTask, SlideEncoderModel, TimmModel


class GigaPath(TimmModel, key="gigapath"):
    is_gated = True
    task = ModelTask.vision
    license = "Apache 2.0 with conditions"
    description = "A whole-slide foundation model for digital pathology"
    commercial = False
    hf_url = "https://huggingface.co/prov-gigapath/prov-gigapath"
    github_url = "https://github.com/prov-gigapath/prov-gigapath"
    paper_url = "https://doi.org/10.1038/s41586-024-07441-w"
    bib_key = "Xu2024-td"
    param_size = "1.13B"
    encode_dim = 1536

    def __init__(self, model_path=None, token=None):
        # Version check
        import timm

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


class GigaPathSlideEncoder(SlideEncoderModel, key="gigapath-slide-encoder"):
    is_gated = True
    task = ModelTask.slide_encoder
    license = "Apache 2.0 with conditions"
    description = "A whole-slide foundation model for digital pathology"
    commercial = False
    hf_url = "https://huggingface.co/prov-gigapath/prov-gigapath"
    github_url = "https://github.com/prov-gigapath/prov-gigapath"
    paper_url = "https://doi.org/10.1038/s41586-024-07441-w"
    bib_key = "Xu2024-td"

    def __init__(self, model_path=None, token=None):
        from huggingface_hub import login

        super().__init__()

        if token is not None:
            login(token)

        try:
            from gigapath.slide_encoder import create_model

            model = create_model(
                "hf_hub:prov-gigapath/prov-gigapath",
                "gigapath_slide_enc12l768d",
                1536,
                local_dir=str(user_cache_path("lazyslide")),
            )
            self.model = model
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Please install gigapath to use this GigaPathSlideEncoder."
                "Try pip install git+https://github.com/prov-gigapath/prov-gigapath"
            )

    def encode_slide(self, tile_embed, coordinates):
        return self.model(tile_embed, coordinates).squeeze()
