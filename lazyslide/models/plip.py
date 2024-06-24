import lazy_loader

from lazyslide.utils import get_torch_device

torch = lazy_loader.load("torch")


# Modified from https://github.com/PathologyFoundation/plip/blob/main/plip.py
class PLIP(torch.nn.Module):
    def __init__(self, model_path=None, auth_token=None):
        from transformers import CLIPModel, CLIPProcessor

        super().__init__()

        if model_path is None:
            model_path = "vinid/plip"
        self.model = CLIPModel.from_pretrained(model_path, use_auth_token=auth_token)
        self.processor = CLIPProcessor.from_pretrained(
            model_path, use_auth_token=auth_token
        )

    def encode_image(self, image, normalize=True):
        inputs = self.processor(images=image, return_tensors="pt")
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
    def __init__(self, model_path=None, auth_token=None):
        from transformers import CLIPVisionModelWithProjection, CLIPProcessor

        super().__init__()

        if model_path is None:
            model_path = "vinid/plip"
        self.model = CLIPVisionModelWithProjection.from_pretrained(
            model_path, use_auth_token=auth_token
        )
        self.processor = CLIPProcessor.from_pretrained(
            model_path, use_auth_token=auth_token
        )

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
