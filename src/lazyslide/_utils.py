from __future__ import annotations

import inspect
import os
from types import FrameType

from rich.console import Console

console = Console()

# Files under this directory are internal to lazyslide. The trailing os.sep keeps
# sibling distributions (e.g. lazyslide_models) from matching the prefix.
_PKG_DIR = os.path.dirname(__file__) + os.sep


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


def chunker(seq, num_workers):
    avg = len(seq) / num_workers
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last) : int(last + avg)])
        last += avg

    return out


def find_stack_level() -> int:
    """Return the ``stacklevel`` of the first caller outside of lazyslide.

    Pass it to :func:`warnings.warn` or :func:`logging.warning` so the message is
    attributed to the user's call site instead of an internal frame.
    """
    # inspect.stack() is slow, walk f_back instead.
    # https://stackoverflow.com/questions/17407119/python-inspect-stack-is-slow
    frame: FrameType | None = inspect.currentframe()
    try:
        n = 0
        while frame is not None and frame.f_code.co_filename.startswith(_PKG_DIR):
            frame = frame.f_back
            n += 1
    finally:
        # See note in
        # https://docs.python.org/3/library/inspect.html#inspect.Traceback
        del frame
    # n is 0 only when currentframe() is unavailable (non-CPython implementations).
    # stacklevel=0 makes logging blame logging/__init__.py, so never return it.
    return max(n, 1)
