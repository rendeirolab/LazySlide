"""
Tests for the model registry and list_models functionality.

These tests verify that the model registry is properly structured and that
the list_models function works correctly, without initializing any models.
"""

from collections.abc import MutableMapping

import pandas as pd
import pytest

from lazyslide.models import MODEL_REGISTRY, list_models
from lazyslide.models._model_registry import ModelCard, ModelTask


def test_model_registry_import():
    """Test that the model registry can be imported."""
    assert MODEL_REGISTRY is not None


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
        assert isinstance(value, ModelCard)


def test_model_card_attributes():
    """Test that ModelCard instances have the expected attributes."""
    # Get the first model card
    first_key = next(iter(MODEL_REGISTRY))
    card = MODEL_REGISTRY[first_key]

    # Check required attributes
    assert hasattr(card, "name")
    assert hasattr(card, "is_gated")
    assert hasattr(card, "model_type")
    assert hasattr(card, "module")

    # Check attribute types
    assert isinstance(card.name, str)
    assert isinstance(card.is_gated, bool)
    assert isinstance(card.model_type, list)
    assert all(isinstance(mt, ModelTask) for mt in card.model_type)


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
            assert task in model.model_type


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
