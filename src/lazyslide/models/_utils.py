from contextlib import contextmanager

import torch


def _fake_class(name, deps, inject=""):
    def init(self, *args, **kwargs):
        raise ImportError(
            f"To use {name}, you need to install {', '.join(deps)}."
            f"{inject}"
            "Please restart the kernel after installation."
        )

    # Dynamically create the class
    new_class = type(name, (object,), {"__init__": init})

    return new_class


@contextmanager
def hf_access(name):
    """
    Context manager for Hugging Face access.
    """
    from huggingface_hub.errors import GatedRepoError

    try:
        yield
    except GatedRepoError as e:
        raise GatedRepoError(
            f"You don't have access to {name}. Please request access to the model on HuggingFace. "
            "After access granted, please login to HuggingFace with huggingface-cli on this machine "
            "with a token that has access to this model. "
            "You may also pass token as an argument in LazySlide, however, this is not recommended."
        ) from e


def get_default_transform():
    """The default transform for the model."""
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.v2 import (
        CenterCrop,
        Compose,
        Normalize,
        Resize,
        ToDtype,
        ToImage,
    )

    transforms = [
        ToImage(),
        Resize(
            size=(224, 224),
            interpolation=InterpolationMode.BICUBIC,
            max_size=None,
            antialias=True,
        ),
        CenterCrop(224),
        ToDtype(dtype=torch.float32, scale=True),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
    return Compose(transforms)
