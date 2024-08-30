import torch


class CONCH(torch.nn.Module):
    def __init__(self, model_path=None, token=None):
        try:
            from conch.open_clip_custom import create_model_from_pretrained
            from conch.open_clip_custom import get_tokenizer
        except ImportError:
            raise ImportError(
                "Conch is not installed. You can install it using "
                "`pip install git+https://github.com/mahmoodlab/CONCH.git`."
            )

        super().__init__()

        if model_path is None:
            model_path = "hf_hub:MahmoodLab/conch"

        self.model, self.processor = create_model_from_pretrained(
            "conch_ViT-B-16", model_path, hf_auth_token=token
        )
        self.tokenizer = get_tokenizer()

    def tokenize(self, text):
        from conch.open_clip_custom import tokenize

        return tokenize(self.tokenizer, text)

    def encode_image(self, image, normalize=True):
        if not isinstance(image, torch.Tensor):
            image = self.processor(image)
        if image.dim() == 3:
            image = image.unsqueeze(0)
        image_feature = self.model.encode_image(
            image, normalize=normalize, proj_contrast=normalize
        )
        return image_feature

    def encode_text(self, text, normalize=True):
        encode_texts = self.tokenize(text)
        text_feature = self.model.encode_text(encode_texts)
        return text_feature

    def forward(self, image):
        return self.encode_image(image)


class CONCHVision(CONCH):
    def forward(self, image):
        return self.encode_image(image, normalize=False)
