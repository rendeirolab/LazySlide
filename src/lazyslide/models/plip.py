# Modified from https://github.com/PathologyFoundation/plip/blob/main/plip.py

import torch


class PLIP(torch.nn.Module):
    def __init__(self, model_path=None, token=None):
        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError:
            raise ImportError(
                "Please install the 'transformers' package to use the PLIP model"
            )

        super().__init__()

        if model_path is None:
            model_path = "vinid/plip"
        self.model = CLIPModel.from_pretrained(model_path, use_auth_token=token)
        self.processor = CLIPProcessor.from_pretrained(model_path, use_auth_token=token)

    def encode_image(self, image, normalize=True):
        if not isinstance(image, torch.Tensor):
            inputs = self.processor(images=image, return_tensors="pt")
        else:
            inputs = {"pixel_values": image}
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        image_features = self.model.get_image_features(**inputs)
        if normalize:
            image_features = torch.nn.functional.normalize(image_features, p=2, dim=-1)
        return image_features

    def encode_text(self, text, normalize=True):
        inputs = self.processor(
            text=text,
            return_tensors="pt",
            max_length=77,
            padding="max_length",
            truncation=True,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        text_features = self.model.get_text_features(**inputs)
        if normalize:
            text_features = torch.nn.functional.normalize(text_features, p=2, dim=-1)
        return text_features

    def forward(self, image):
        image_features = self.encode_image(image, normalize=True)
        return image_features


class PLIPVision(torch.nn.Module):
    def __init__(self, model_path=None, token=None):
        try:
            from transformers import CLIPVisionModelWithProjection, CLIPProcessor
        except ImportError:
            raise ImportError(
                "Please install the 'transformers' package to use the PLIP model"
            )

        super().__init__()

        if model_path is None:
            model_path = "vinid/plip"
        self.model = CLIPVisionModelWithProjection.from_pretrained(
            model_path, use_auth_token=token
        )
        self.processor = CLIPProcessor.from_pretrained(model_path, use_auth_token=token)

    def encode_image(self, image, normalize=False):
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        image_features = self.model.get_image_features(**inputs)
        if normalize:
            image_features = torch.nn.functional.normalize(image_features, p=2, dim=-1)
        return image_features

    def forward(self, image):
        image_features = self.encode_image(image, normalize=False)
        return image_features
