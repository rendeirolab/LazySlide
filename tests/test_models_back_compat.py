"""
Tests for the model backward compatibility.
"""

import importlib
import sys

import pytest

import lazyslide


def test_backward_compat_module_aliases(monkeypatch):
    """Legacy lazyslide.models.* imports resolve to lazyslide_models modules, and warn."""
    # Import the shim afresh, so it runs (and warns) even if another test loaded it
    for name in [m for m in sys.modules if m.split(".")[:2] == ["lazyslide", "models"]]:
        monkeypatch.delitem(sys.modules, name)
    if "models" in vars(lazyslide):
        monkeypatch.delitem(vars(lazyslide), "models")

    with pytest.warns(FutureWarning, match="`lazyslide.models` is deprecated"):
        base_module = importlib.import_module("lazyslide.models.base")
    compat_base_module = importlib.import_module("lazyslide_models.base")
    hibou_module = importlib.import_module("lazyslide.models.vision.hibou")
    compat_hibou_module = importlib.import_module("lazyslide_models.vision.hibou")

    assert base_module is compat_base_module
    assert hibou_module is compat_hibou_module
    assert sys.modules["lazyslide.models.base"] is compat_base_module
    assert sys.modules["lazyslide.models.vision.hibou"] is compat_hibou_module
