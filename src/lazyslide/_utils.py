from __future__ import annotations

import importlib
import inspect
import os
import warnings
from types import FrameType

from rich.console import Console

console = Console()

# Files under this directory are internal to lazyslide. The trailing os.sep keeps
# sibling distributions (e.g. lazyslide_models) from matching the prefix.
_PKG_DIR = os.path.dirname(__file__) + os.sep
# The import machinery: a warning raised while a lazyslide module is imported
# (e.g. the deprecated lazyslide.models) must not be blamed on importlib.
_IMPORTLIB_DIR = os.path.dirname(importlib.__file__) + os.sep


def get_torch_device():
    """Automatically get the torch device"""
    import torch

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


def default_pbar(disable=False):
    """Get the default progress bar"""
    from rich.progress import (
        BarColumn,
        Progress,
        TaskProgressColumn,
        TextColumn,
        TimeRemainingColumn,
    )

    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=30),
        TaskProgressColumn(),
        TimeRemainingColumn(compact=True, elapsed_when_finished=True),
        disable=disable,
        console=console,
        transient=True,
    )


def find_stack_level() -> int:
    """Return the ``stacklevel`` of the first caller outside of lazyslide.

    Pass it to :func:`warnings.warn` or :func:`logging.warning` so the message is
    attributed to the user's call site instead of an internal frame. Frames of the
    import machinery are passed over too, so a warning raised at import time names
    the user's import statement or attribute access.
    """
    # inspect.stack() is slow, walk f_back instead.
    # https://stackoverflow.com/questions/17407119/python-inspect-stack-is-slow
    frame: FrameType | None = inspect.currentframe()
    try:
        n = 0
        while frame is not None:
            filename = frame.f_code.co_filename
            # warnings and logging skip importlib's bootstrap frames themselves,
            # without counting them, so they must not be counted here either
            if not ("importlib" in filename and "_bootstrap" in filename):
                if not filename.startswith((_PKG_DIR, _IMPORTLIB_DIR)):
                    break
                n += 1
            frame = frame.f_back
    finally:
        # See note in
        # https://docs.python.org/3/library/inspect.html#inspect.Traceback
        del frame
    # n is 0 only when currentframe() is unavailable (non-CPython implementations).
    # stacklevel=0 makes logging blame logging/__init__.py, so never return it.
    return max(n, 1)


def warn_deprecated(msg: str) -> None:
    """Emit ``msg`` as a :class:`FutureWarning` attributed to the user's call site."""
    warnings.warn(msg, FutureWarning, stacklevel=find_stack_level())


def deprecated_alias(old_name: str, old, new_name: str, new, default=None):
    """Resolve a renamed keyword argument.

    Returns the value to use for ``new_name``. ``old`` is the value passed under the
    deprecated name (``None`` when not given) and ``default`` is the default of
    ``new_name``, used to tell whether ``new_name`` was passed too.
    """
    if old is None:
        return new
    if new != default:
        raise TypeError(
            f"`{old_name}` is a deprecated alias of `{new_name}`; "
            f"pass only `{new_name}`."
        )
    warn_deprecated(
        f"`{old_name}` is deprecated since v0.13.0 and will be removed in v0.14.0; "
        f"use `{new_name}`."
    )
    return old
