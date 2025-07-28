from __future__ import annotations

import json
from collections.abc import MutableMapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Iterator, List, Type

import pandas as pd

from . import multimodal, segmentation, tile_prediction, vision
from .base import ModelBase


class ModelTask(Enum):
    vision = "vision"
    segmentation = "segmentation"
    multimodal = "multimodal"
    tile_prediction = "tile_prediction"
    slide_encoder = "slide_encoder"


@dataclass
class ModelCard:
    name: str
    is_gated: bool
    model_type: List[ModelTask]
    module: Type[ModelBase]
    commercial: bool
    github_url: str = None
    hf_url: str = None
    paper_url: str = None
    description: str = None
    keys: List[str] = None
    bib_key: str = None
    license: str | List[str] = None
    license_url: str = None
    param_size: str = None
    encode_dim: int = None

    def _repr_html_(self) -> str:
        """Return an HTML representation of the model card for Jupyter notebooks.

        This method is automatically called by Jupyter notebooks to display
        the model card as an HTML table.

        Returns
        -------
        str
            HTML representation of the model card.
        """
        # Create a styled HTML representation
        html = [
            '<div style="border: 1px solid #ddd; border-radius: 8px; padding: 15px; margin: 10px 0; '
            'background-color: #ffffff; box-shadow: 0 2px 4px rgba(0,0,0,0.1); max-width: fit-content;">',
            f'<h3 style="margin: 0 0 15px 0; text-align: center; color: #2c3e50;">{self.name}</h3>',
            '<ul style="list-style: none; padding: 0; margin: 0;">',
            '<li style="margin-bottom: 8px;">',
            f'<span style="font-weight: bold;">Model type:</span> {"; ".join([m.value for m in self.model_type])}</li>',
        ]

        # Add status with icon
        status_icon = "🔒" if self.is_gated else "✅"
        status_text = "Gated" if self.is_gated else "Open"
        html.append(
            f'<li style="margin-bottom: 8px;"><span style="font-weight: bold;">'
            f"Status:</span> {status_icon} {status_text}</li>"
        )

        # Add description if available
        if self.description:
            html.append(
                f'<li style="margin-bottom: 8px;"><span style="font-weight: bold;">'
                f"Description:</span> {self.description}</li>"
            )

        # Add links section if any links are available
        links = []
        button_style = (
            "display: inline-block; padding: 6px 12px; margin: 0 6px 6px 0; border-radius: 4px; "
            "text-decoration: none; color: white; background-color: #3498db;"
        )

        if self.github_url:
            links.append(
                f'<a href="{self.github_url}" target="_blank" style="{button_style}">GitHub</a>'
            )
        if self.hf_url:
            links.append(
                f'<a href="{self.hf_url}" target="_blank" style="{button_style}">Hugging Face</a>'
            )
        if self.paper_url:
            links.append(
                f'<a href="{self.paper_url}" target="_blank" style="{button_style}">Paper</a>'
            )

        if links:
            html.append(f'<li style="margin-bottom: 8px;">{"".join(links)}</li>')

        # Close the HTML
        html.append("</ul></div>")

        return "".join(html)

    def __post_init__(self):
        try:
            inject_doc = self._doc()
            origin_doc = self.module.__doc__
            if origin_doc is None:
                origin_doc = ""
            else:
                origin_doc = f"\n\n{origin_doc}"
            self.module.__doc__ = f"{inject_doc}{origin_doc}"
        except AttributeError:
            # If the module does not have a __doc__ attribute, skip the injection
            pass

        if self.keys is None:
            self.keys = [self.name.lower()]

    def _doc(self):
        skeleton = (
            ":octicon:`lock;1em;sd-text-danger;` "
            if self.is_gated
            else ":octicon:`check-circle-fill;1em;sd-text-success;` "
        )
        if self.hf_url is not None:
            skeleton += f":bdg-link-primary-line:`🤗Hugging Face <{self.hf_url}>` "
        if self.github_url is not None:
            skeleton += f":bdg-link-primary-line:`GitHub <{self.github_url}>` "
        if self.paper_url is not None:
            skeleton += f":bdg-link-primary-line:`Paper <{self.paper_url}>` "
        if self.param_size is not None:
            skeleton += f":bdg-info-line:`Params: {self.param_size}` "
        if self.encode_dim:
            skeleton += f":bdg-info-line:`{self.encode_dim} features` "
        if self.license is not None:
            if isinstance(self.license, list):
                license_str = "; ".join(self.license)
            else:
                license_str = self.license
            if self.license_url is not None:
                skeleton += f":bdg-link-light:`{license_str} <{self.license_url}>` "
            else:
                skeleton += f":bdg-light:`{license_str}` "
        if self.bib_key is not None:
            skeleton += f":cite:p:`{self.bib_key}` "
        if self.description is not None:
            skeleton += f"\n{self.description}"

        return skeleton


class ModelRegistry(MutableMapping):
    """A registry for models that behaves like a dictionary but with enhanced representation.

    This class implements the MutableMapping interface and provides additional
    functionality for displaying the registry as a DataFrame and HTML.
    """

    def __init__(self):
        """Initialize an empty model registry."""
        self._data: Dict[str, ModelCard] = {}

    def __getitem__(self, key: str) -> ModelCard:
        """Get a model card by key."""
        return self._data[key]

    def __setitem__(self, key: str, value: ModelCard) -> None:
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
        data = []
        for key, card in self._data.items():
            data.append(
                {
                    "key": key,
                    "name": card.name,
                    "model_type": "; ".join([m.value for m in card.model_type]),
                    "is_gated": card.is_gated,
                    "github_url": card.github_url,
                    "hf_url": card.hf_url,
                    "paper_url": card.paper_url,
                    "description": card.description,
                    "bib_key": card.bib_key,
                }
            )
        return pd.DataFrame(data)

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
        if not self._data:
            return f"<p>{type(self).__qualname__}()</p>"

        # Get the DataFrame
        used_cols = [
            "is_gated",
            "key",
            "model_type",
            "github_url",
            "hf_url",
            "paper_url",
        ]
        display_cols = [
            "Gating Status",
            "Key",
            "Model Type",
            "GitHub",
            "HuggingFace",
            "Paper",
        ]
        df = self.to_dataframe().loc[:, used_cols]

        def format_is_gated(value):
            # Convert boolean is_gated to emoji icon
            return "🔒" if value else "✅"

        # Define a function to style rows based on is_gated value
        def style_rows(row):
            bg_color = (
                "pink" if row["is_gated"] else "#e8f5e9"
            )  # Red for gated, green for open
            return [f"background-color: {bg_color}"] * len(row)

        # Define a function to format URL columns as link buttons
        def format_url(value):
            if value is None:
                return ""
            return f'<a href="{value}" target="_blank">Link</a>'

        # Create a styler object
        styler = (
            df.style
            # Apply row styling based on is_gated
            .apply(style_rows, axis=1)
            # Format URL columns as link buttons
            .format(
                {
                    "hf_url": format_url,
                    "paper_url": format_url,
                    "github_url": format_url,
                    "is_gated": format_is_gated,
                },
                escape="html",
            )
            # Rename the column name
            .relabel_index(display_cols, axis="columns")
            # Hide the index
            .hide(axis="index")
        )

        return styler.to_html()


MODEL_REGISTRY = ModelRegistry()

with open(f"{Path(__file__).parent}/model_registry.json", "r") as f:
    MODEL_DB = json.load(f)

_modules = {
    ModelTask.vision: vision,
    ModelTask.segmentation: segmentation,
    ModelTask.multimodal: multimodal,
    ModelTask.tile_prediction: tile_prediction,
}

for rec in MODEL_DB:
    try:
        model_type = rec["model_type"]
        if isinstance(model_type, str):
            model_type = [model_type]
        model_type = [ModelTask(m) for m in model_type]

        # Get module
        mod, cls = rec["module"].split(".")
        if mod not in ModelTask.__members__:
            raise ValueError(
                f"Invalid module task '{mod}' in record '{rec['name']}'. "
                f"Valid tasks are: {', '.join(ModelTask.__members__.keys())}."
            )
        module = getattr(_modules[ModelTask(mod)], cls)

        card = ModelCard(
            name=rec["name"],
            is_gated=rec["is_gated"],
            model_type=model_type,
            module=module,
            commercial=rec["commercial"],
            github_url=rec.get("github_url"),
            hf_url=rec.get("hf_url"),
            paper_url=rec.get("paper_url"),
            description=rec.get("description"),
            bib_key=rec.get("bib_key"),
            license=rec.get("license"),
            license_url=rec.get("license_url"),
            param_size=rec.get("param_size"),
            encode_dim=rec.get("encode_dim"),
        )
        keys = rec["keys"]
        for key in keys:
            MODEL_REGISTRY[key] = card
    except Exception as e:
        raise ImportError(f"Failed to parse {rec['name']} in model registry") from e


def list_models(task: ModelTask = None):
    """List all available models.

    If you want to get models for feature extraction,
    you can use task='vision' or task='multimodal'.

    Parameters
    ----------
    task : {'vision', 'segmentation', 'multimodal', 'tile_prediction'}, default: None
        The task to filter the models. If None, return all models.

    Returns
    -------
    list
        A list of model names.

    """
    if task is None:
        return list(MODEL_REGISTRY.keys())
    if task is not None:
        task = ModelTask(task)
        if task in ModelTask:
            return [
                name
                for name, model in MODEL_REGISTRY.items()
                if task in model.model_type
            ]
        else:
            raise ValueError(
                f"Unknown task: {task}. "
                "Available tasks are: vision, segmentation, multimodal and tile_prediction."
            )
