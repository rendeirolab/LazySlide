from __future__ import annotations

from collections.abc import MutableMapping
from typing import TYPE_CHECKING, Dict, Iterator

import pandas as pd

from ._repr import model_registry_repr_html

if TYPE_CHECKING:
    from .base import ModelBase


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
                    "is_gated": model.is_gated,
                    "github_url": model.github_url,
                    "hf_url": model.hf_url,
                    "paper_url": model.paper_url,
                    "description": model.description,
                    "bib_key": model.bib_key,
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
