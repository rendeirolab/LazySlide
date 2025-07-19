# Implement lazy loading for submodules
import importlib
from types import ModuleType


class LazyLoader(ModuleType):
    """Lazily import a module, only when it is needed."""

    def __init__(self, name, package=None):
        super().__init__(name)
        self._name = name
        self._package = package
        self._module = None

    def __getattr__(self, name):
        if self._module is None:
            self._module = importlib.import_module(self._name, self._package)
            # Update this object's dict so that if someone keeps a reference to the
            # LazyLoader, lookups are efficient (__getattr__ is only called on lookups
            # that fail).
            self.__dict__.update(self._module.__dict__)
        return getattr(self._module, name)

    def __dir__(self):
        """Return the list of names of module attributes."""
        if self._module is None:
            self._module = importlib.import_module(self._name, self._package)
        return dir(self._module)

    def __setattr__(self, name, value):
        """Set an attribute on the module."""
        if name in ["_name", "_package", "_module"]:
            # Set attributes of the LazyLoader instance
            super().__setattr__(name, value)
        else:
            # Set attributes on the actual module
            if self._module is None:
                self._module = importlib.import_module(self._name, self._package)
            setattr(self._module, name, value)

    def __delattr__(self, name):
        """Delete an attribute from the module."""
        if name in ["_name", "_package", "_module"]:
            # Delete attributes of the LazyLoader instance
            super().__delattr__(name)
        else:
            # Delete attributes from the actual module
            if self._module is None:
                self._module = importlib.import_module(self._name, self._package)
            delattr(self._module, name)

    def __reduce__(self):
        return importlib.import_module, (self.__name__,)
