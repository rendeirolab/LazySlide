from __future__ import annotations

import textwrap
import warnings
from collections.abc import MutableMapping
from typing import TYPE_CHECKING, Dict, Iterator

import pandas as pd

from .._utils import find_stack_level
from ._repr import model_doc, model_registry_repr_html

if TYPE_CHECKING:
    from .base import ModelBase, ModelTask


class ModelRegistry(MutableMapping):
    """A registry for models that behaves like a dictionary but with enhanced representation.

    This class implements the MutableMapping interface and provides additional
    functionality for displaying the registry as a DataFrame and HTML.
    """

    def __init__(self):
        """Initialize an empty model registry."""
        self._data: Dict[str, type[ModelBase]] = {}

    def __getitem__(self, key: str) -> type[ModelBase]:
        """Get a model card by key."""
        model = self._data.get(key)
        if model is None:
            raise KeyError(f"Cannot find model '{key}' in registry.")
        return model

    def __setitem__(self, key: str, value: type[ModelBase]) -> None:
        """Add a model card to the registry."""
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        """Remove a model card from the registry."""
        del self._data[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate over the keys in the registry."""
        return iter(self._data)

    def __len__(self) -> int:
        """Get the number of models in the registry."""
        return len(self._data)

    def __contains__(self, key: str) -> bool:
        """Check if a model card exists in the registry."""
        return key in self._data

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the registry to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing information about all models in the registry.
        """
        data = {}
        for key, model in self._data.items():
            if isinstance(model.task, list):
                task = [m.value for m in model.task]
            else:
                task = [model.task.value]
            if model in data:
                data[model]["key"].append(key)
            else:
                data[model] = {
                    "key": [key],
                    "name": model.__name__,
                    "model_type": "; ".join(task),
                    "is_gated": getattr(model, "is_gated", None),
                    "github_url": getattr(model, "github_url", None),
                    "hf_url": getattr(model, "hf_url", None),
                    "paper_url": getattr(model, "paper_url", None),
                    "description": getattr(model, "description", None),
                    "bib_key": getattr(model, "bib_key", None),
                }
        for k, record in data.items():
            record["key"] = "; ".join(record["key"])
        return pd.DataFrame(data.values()).sort_values(
            ["model_type", "key"], ascending=[False, True]
        )

    def __repr__(self) -> str:
        """Return a string representation of the registry as a DataFrame."""
        if not self._data:
            return f"{type(self).__qualname__}()"
        return self.to_dataframe().to_string()

    def _repr_html_(self) -> str:
        """Return an HTML representation of the registry for Jupyter notebooks.

        This method is automatically called by Jupyter notebooks to display
        the registry as an HTML table.

        Returns
        -------
        str
            HTML representation of the registry.
        """
        return model_registry_repr_html(self)


# Global instance
MODEL_REGISTRY = ModelRegistry()


def register(
    key: str | list[str],
    task: ModelTask | list[ModelTask] = None,
    is_gated: bool = False,
    license: str | list[str] = None,
    license_url: str | list[str] = None,
    description: str = None,
    commercial: bool = None,
    github_url: str = None,
    hf_url: str = None,
    paper_url: str = None,
    bib_key: str = None,
    param_size: int | str = None,
    encode_dim: int = None,
    vision_encoder: str = None,
    flops: int | str = None,
    **information,
):
    """Register a model class with additional information."""

    def decorator(cls: type[ModelBase]):
        # Allow multiple names
        keys = key if isinstance(key, list) else [key]
        for k in keys:
            if k in MODEL_REGISTRY:
                warnings.warn(
                    f"Model name {k} already registered, consider using another name.",
                    stacklevel=find_stack_level(),
                )
            MODEL_REGISTRY[k] = cls
        if task is None:
            raise ValueError("task must be specified when registering a model class.")
        # Set model attributes
        cls.task = task
        cls.is_gated = is_gated
        cls.license = license
        cls.license_url = license_url
        cls.description = description
        cls.commercial = commercial
        cls.github_url = github_url
        cls.hf_url = hf_url
        cls.paper_url = paper_url
        cls.bib_key = bib_key
        cls.param_size = param_size
        cls.encode_dim = encode_dim
        cls.vision_encoder = vision_encoder
        cls.flops = flops

        # Set any additional information
        for info_key, info_value in information.items():
            setattr(cls, info_key, info_value)

        old_doc = cls.__doc__
        if old_doc is None:
            old_doc = ""
        else:
            old_doc = textwrap.dedent(old_doc)
        inject_doc = model_doc(cls)
        cls.__doc__ = inject_doc + "\n" + old_doc

        return cls

    return decorator
