import torch

from .._utils import hf_access
from ..base import ImageTextModel, ModelTask


class MUSK(ImageTextModel, key="musk"):
    is_gated = True
    task = ModelTask.multimodal
    license = "CC-BY-NC-ND-4.0"
    description = "A Vision-Language Foundation Model for Precision Oncology"
    commercial = False
    hf_url = "https://huggingface.co/xiangjx/musk"
    github_url = "https://github.com/lilab-stanford/MUSK"
    paper_url = "https://doi.org/10.1038/s41591-024-02856-4"
    bib_key = "Xiang2025-fd"
    param_size = "675.2M"
    encode_dim = 1024

    def __init__(
        self,
        model_path=None,
        token=None,
        multiscale_augmentation=False,
        with_head=True,
    ):
        from huggingface_hub import hf_hub_download
        from timm.models import create_model
        from transformers import XLMRobertaTokenizer

        try:
            from musk import modeling, utils
        except ImportError:
            raise ImportError(
                "MUSK is not installed. You can install it using "
                "`pip install git+https://github.com/lilab-stanford/MUSK`."
            )

        self.ms_aug = multiscale_augmentation
        self.with_head = with_head

        if model_path is None:
            model_path = "hf_hub:xiangjx/musk"

        with hf_access(model_path):
            model = create_model("musk_large_patch16_384")
            utils.load_model_and_may_interpolate(
                "hf_hub:xiangjx/musk", model, "model|module", ""
            )
            model.eval()
            self.model = model
            token_file = hf_hub_download(
                repo_id="xiangjx/musk", filename="tokenizer.spm"
            )
            self.tokenizer = XLMRobertaTokenizer(token_file)

    def get_transform(self):
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.v2 import (
            CenterCrop,
            Compose,
            Normalize,
            Resize,
            ToDtype,
            ToImage,
        )

        return Compose(
            [
                ToImage(),
                Resize(384, interpolation=InterpolationMode.BICUBIC, antialias=True),
                CenterCrop(384),
                ToDtype(dtype=torch.float32, scale=True),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @torch.inference_mode()
    def encode_image(self, image):
        # Move image to the same device as the model
        # Get the model device
        image_feature = self.model(
            image, with_head=self.with_head, out_norm=False, ms_aug=self.ms_aug
        )[0]
        return image_feature

    @torch.inference_mode()
    def encode_text(self, text):
        from musk import utils

        # Move tokenized text to the same device as the model
        # Get the model device
        try:
            device = next(self.model.parameters()).device
        except Exception:
            device = torch.device("cpu")

        if isinstance(text, str):
            text = [text]

        text_ids = []
        paddings = []
        for txt in text:
            txt_ids, pad = utils.xlm_tokenizer(txt, self.tokenizer, max_len=100)
            text_ids.append(torch.tensor(txt_ids).unsqueeze(0))
            paddings.append(torch.tensor(pad).unsqueeze(0))

        text_ids = torch.cat(text_ids).to(device)
        paddings = torch.cat(paddings).to(device)

        text_feature = self.model(
            text_description=text_ids,
            padding_mask=paddings,
            with_head=self.with_head,
            out_norm=True,
        )[1]
        return text_feature
