"""
Tests for the model registry and list_models functionality.

These tests verify that the model registry is properly structured and that
the list_models function works correctly, without initializing any models.
"""

import importlib
import sys
from collections.abc import MutableMapping

import pandas as pd
import pytest

from lazyslide.models import MODEL_REGISTRY, ModelBase, ModelTask, list_models


def test_model_registry_import():
    """Test that the model registry can be imported."""
    assert MODEL_REGISTRY is not None


def test_backward_compat_module_aliases():
    """Legacy lazyslide.models.* imports should resolve to lazyslide_models modules."""
    base_module = importlib.import_module("lazyslide.models.base")
    compat_base_module = importlib.import_module("lazyslide_models.base")
    registry_module = importlib.import_module("lazyslide.models._model_registry")
    compat_registry_module = importlib.import_module("lazyslide_models._model_registry")
    hibou_module = importlib.import_module("lazyslide.models.vision.hibou")
    compat_hibou_module = importlib.import_module("lazyslide_models.vision.hibou")

    assert base_module is compat_base_module
    assert registry_module is compat_registry_module
    assert hibou_module is compat_hibou_module
    assert sys.modules["lazyslide.models.base"] is compat_base_module
    assert sys.modules["lazyslide.models.vision.hibou"] is compat_hibou_module


def test_model_registry_type():
    """Test that MODEL_REGISTRY is a MutableMapping."""
    assert isinstance(MODEL_REGISTRY, MutableMapping)


def test_model_registry_not_empty():
    """Test that MODEL_REGISTRY is not empty."""
    assert len(MODEL_REGISTRY) > 0


def test_model_registry_keys():
    """Test that MODEL_REGISTRY keys are strings."""
    for key in MODEL_REGISTRY:
        assert isinstance(key, str)


def test_model_registry_values():
    """Test that MODEL_REGISTRY values are ModelCard instances."""
    for value in MODEL_REGISTRY.values():
        assert issubclass(value, ModelBase)


def test_model_attributes():
    """Test that ModelCard instances have the expected attributes."""
    # Get the first model card
    first_key = next(iter(MODEL_REGISTRY))
    card = MODEL_REGISTRY[first_key]

    # Check required attributes
    assert hasattr(card, "name")
    assert hasattr(card, "is_gated")
    assert hasattr(card, "task")

    # Check attribute types
    assert isinstance(card.is_gated, bool)
    if isinstance(card.task, ModelTask):
        task = [card.task]
    else:
        task = card.task
    for t in task:
        assert isinstance(t, ModelTask)


def test_model_registry_to_dataframe():
    """Test that MODEL_REGISTRY.to_dataframe() returns a DataFrame."""
    df = MODEL_REGISTRY.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "name" in df.columns
    assert "is_gated" in df.columns
    assert "model_type" in df.columns


def test_list_models_all():
    """Test that list_models() returns all models."""
    models = list_models()
    assert isinstance(models, list)
    assert len(models) > 0
    assert len(models) == len(MODEL_REGISTRY)
    assert set(models) == set(MODEL_REGISTRY.keys())


def test_list_models_by_task():
    """Test that list_models(task) returns models filtered by task."""
    for task in ModelTask:
        models = list_models(task)
        assert isinstance(models, list)

        # Verify that all returned models have the specified task
        for model_key in models:
            model = MODEL_REGISTRY[model_key]
            if isinstance(model.task, ModelTask):
                tasks = [model.task]
            else:
                tasks = model.task
            assert task in tasks


def test_list_models_invalid_task():
    """Test that list_models() raises ValueError for invalid task."""
    with pytest.raises(ValueError):
        list_models("invalid_task")


def test_model_registry_repr():
    """Test that MODEL_REGISTRY has a string representation."""
    repr_str = repr(MODEL_REGISTRY)
    assert isinstance(repr_str, str)
    assert len(repr_str) > 0


def test_model_registry_html_repr():
    """Test that MODEL_REGISTRY has an HTML representation."""
    assert hasattr(MODEL_REGISTRY, "_repr_html_")
    html = MODEL_REGISTRY._repr_html_()
    assert isinstance(html, str)
    assert len(html) > 0
    assert "<table" in html
    assert "</table>" in html
