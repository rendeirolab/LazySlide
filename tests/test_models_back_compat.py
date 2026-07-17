"""
Tests for the model backward compatibility.
"""

import importlib
import sys


def test_backward_compat_module_aliases():
    """Legacy lazyslide.models.* imports should resolve to lazyslide_models modules."""
    base_module = importlib.import_module("lazyslide.models.base")
    compat_base_module = importlib.import_module("lazyslide_models.base")
    hibou_module = importlib.import_module("lazyslide.models.vision.hibou")
    compat_hibou_module = importlib.import_module("lazyslide_models.vision.hibou")

    assert base_module is compat_base_module
    assert hibou_module is compat_hibou_module
    assert sys.modules["lazyslide.models.base"] is compat_base_module
    assert sys.modules["lazyslide.models.vision.hibou"] is compat_hibou_module
