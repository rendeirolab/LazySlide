from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Protocol,
    Self,
    Tuple,
    runtime_checkable,
)

import torch
from torch.utils.flop_counter import FlopCounterMode

from ._repr import model_repr_html
from ._utils import get_default_transform, hf_access

if TYPE_CHECKING:
    from numpy.typing import ArrayLike
    from wsidata import TileSpec


class ModelTask(Enum):
    vision = "vision"
    segmentation = "segmentation"
    multimodal = "multimodal"
    slide_encoder = "slide_encoder"
    tile_prediction = "tile_prediction"
    spatial_transcriptomics = "spatial_transcriptomics"
    style_transfer = "style_transfer"
    cv_feature = "cv_feature"
    image_generation = "image_generation"


@runtime_checkable
class ModelBaseProtocol(Protocol):
    model: Any
    name: str

    def get_transform(self) -> Callable | None: ...

    def to(self, device) -> Self: ...

    def try_compile(self, **compile_kws: Any): ...


@runtime_checkable
class ImageModelProtocol(ModelBaseProtocol, Protocol):
    def get_transform(self) -> Callable: ...

    def encode_image(self, image, *args, **kwargs) -> ArrayLike: ...


@runtime_checkable
class ViTModelProtocol(ModelBaseProtocol, Protocol):
    grid_size: Tuple[int, int]
    patch_size: Tuple[int, int]
    num_prefix_tokens: int

    def encode_image_dense(self, image, *args, **kwargs) -> ArrayLike: ...


@runtime_checkable
class ImageTextModelProtocol(ImageModelProtocol, Protocol):
    def encode_image(self, image, *args, **kwargs) -> ArrayLike: ...

    def encode_text(self, text, *args, **kwargs) -> ArrayLike: ...


@runtime_checkable
class SegmentationModelProtocol(ModelBaseProtocol, Protocol):
    def predict(self, image, *args, **kwargs) -> dict[str, Any]: ...

    def supported_formats(self) -> tuple[str]: ...


@runtime_checkable
class TilePredictionProtocol(ModelBaseProtocol, Protocol):
    def predict(self, image, *args, **kwargs) -> Any: ...


@runtime_checkable
class StyleTransferModelProtocol(ModelBaseProtocol, Protocol):
    def predict(self, image): ...

    def get_channel_names(self): ...


@runtime_checkable
class ImageGenerationModelProtocol(ModelBaseProtocol, Protocol):
    def generate(self, *args, **kwargs): ...

    def generate_conditionally(self, *args, **kwargs): ...


class ModelBase(ABC):
    model: Any

    def _repr_html_(self) -> str:
        return model_repr_html(self)

    def get_transform(self) -> Callable | None:
        return None

    def to(self, device) -> Self:
        self.model.to(device)
        return self

    def try_compile(self, **compile_kws: Any):
        try:
            self.model = torch.compile(self.model, **compile_kws)
        except Exception:  # noqa
            pass

    def estimate_param_size(self) -> int | None:
        """Count the number of parameters in a model."""
        model = self.model
        if not isinstance(model, torch.nn.Module):
            try:
                # If it's a Coco model, get the underlying PyTorch model
                model = model.model
            except (AttributeError, TypeError):
                return None
        return sum(p.numel() for p in model.parameters())

    def _resolve_method(
        self, model: torch.nn.Module, method: str
    ) -> tuple[Any, torch.nn.Module] | None:
        """Resolve method path and return (callable, target_model) for FLOPS counting."""
        if "." in method:
            parts = method.split(".")
            obj, target = model, model
            for part in parts[:-1]:
                obj = getattr(obj, part, None)
                if obj is None:
                    return None
                if isinstance(obj, torch.nn.Module):
                    target = obj
            method_obj = getattr(obj, parts[-1], None)
            return (method_obj, target) if method_obj else None

        method_obj = getattr(model, method, None) or getattr(self, method, None)
        return (method_obj, model) if method_obj else None

    def estimate_flops(
        self, *args: Any, method: str = "forward", **kwargs: Any
    ) -> int | None:
        """Count the number of flops in a model."""
        model = self.model
        if not isinstance(model, torch.nn.Module):
            try:
                model = model.model
            except (AttributeError, TypeError):
                return None
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        result = self._resolve_method(model, method)
        if result is None:
            return None

        method_obj, target = result
        is_training = model.training
        model.eval()
        with FlopCounterMode(target, display=False, depth=None) as flop_counter:
            method_obj(*args, **kwargs)
        model.train(is_training)
        return flop_counter.get_total_flops()

    @property
    def name(self):
        return self.__class__.__name__

    @classmethod
    def check_input_tile(cls, tile_spec: "TileSpec") -> bool:
        """
        A helper function to check if the input tile size is valid.

        Return True if the input tile size is valid. And the model will be executed.
        Add a warning here if the input is not optimal but still can be executed.
        """
        return True


class ImageModel(ModelBase):
    def get_transform(self) -> Callable:
        import torch
        from torchvision.transforms.v2 import (
            Compose,
            Normalize,
            Resize,
            ToDtype,
            ToImage,
        )

        return Compose(
            [
                ToImage(),
                ToDtype(dtype=torch.float32, scale=True),
                Resize(size=(224, 224), antialias=False),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    @abstractmethod
    def encode_image(self, image: ArrayLike, *args, **kwargs) -> ArrayLike:
        raise NotImplementedError

    def __call__(self, image: ArrayLike, *args, **kwargs):
        return self.encode_image(image)


class TimmModel(ModelBase):
    def __init__(self, name, token=None, compile=False, compile_kws=None, **kwargs):
        import timm
        from huggingface_hub import login

        if token is not None:
            login(token)

        default_kws = {"pretrained": True, "num_classes": 0}
        default_kws.update(kwargs)

        with hf_access(name):
            self.model = timm.create_model(name, **default_kws)
            try:
                self.model.eval()
            except AttributeError:
                pass

        if compile:
            self.try_compile(**(compile_kws or {}))

        if hasattr(self.model, "default_cfg"):
            self.img_size = self.model.default_cfg.get("input_size", (3, 224, 224))[1:]
        else:
            self.img_size = (224, 224)

    def get_transform(self):
        return get_default_transform(self.img_size)

    @torch.inference_mode()
    def encode_image(self, image: torch.Tensor, *args, **kwargs) -> ArrayLike:
        return self.model(image)


class TimmViTModel(TimmModel):
    def __init__(self, name, token=None, compile=False, compile_kws=None, **kwargs):
        super().__init__(
            name, token=token, compile=compile, compile_kws=compile_kws, **kwargs
        )
        from timm.models import VisionTransformer

        self.is_timm_vit = isinstance(self.model, VisionTransformer)
        if not self.is_timm_vit:
            raise ValueError(f"Model {name} is not a timm VisionTransformer")

        patch_embed = self.model.patch_embed
        self.img_size: Tuple[int, int] = patch_embed.img_size
        self.patch_size: Tuple[int, int] = patch_embed.patch_size
        self.grid_size: Tuple[int, int] = patch_embed.grid_size
        self.num_prefix_tokens: int = self.model.num_prefix_tokens

    @torch.inference_mode()
    def encode_image_dense(self, image: torch.Tensor, *args, **kwargs) -> ArrayLike:
        return self.model.forward_features(image)


class SlideEncoderModel(ModelBase):
    @abstractmethod
    def encode_slide(self, embeddings, coords=None, *args, **kwargs) -> ArrayLike:
        raise NotImplementedError


class ImageTextModel(ImageModel):
    @abstractmethod
    def encode_text(self, text, *args, **kwargs) -> ArrayLike:
        raise NotImplementedError

    def tokenize(self, text, *args, **kwargs):
        raise NotImplementedError


class SegmentationModel(ModelBase):
    probability_map_key = "probability_map"
    instance_map_key = "instance_map"
    class_map_key = "class_map"
    token_map_key = "token_map"

    def get_transform(self):
        import torch
        from torchvision.transforms.v2 import Compose, Normalize, ToDtype, ToImage

        return Compose(
            [
                ToImage(),
                ToDtype(dtype=torch.float32, scale=True),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    @abstractmethod
    def segment(self, image) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def supported_output(self):
        return (
            "probability_map",
            "instance_map",
            "class_map",
            "token_map",
        )

    @staticmethod
    def get_classes():
        return None


class TilePredictionModel(ModelBase):
    @abstractmethod
    def predict(self, image):
        """The output should always be a dict of numpy arrays
        to allow multiple outputs.
        """
        raise NotImplementedError


class StyleTransferModel(ModelBase):
    @abstractmethod
    def predict(self, image):
        raise NotImplementedError

    @abstractmethod
    def get_channel_names(self):
        raise NotImplementedError


class ImageGenerationModel(ModelBase):
    def generate(self, *args, **kwargs):
        raise NotImplementedError

    def generate_conditionally(self, *args, **kwargs):
        raise NotImplementedError
