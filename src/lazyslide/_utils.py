from __future__ import annotations

import inspect
import os
from functools import wraps
from types import FrameType

from rich.console import Console

console = Console()


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
    """
    Find the first place in the stack that is not inside pandas
    (tests notwithstanding).
    """

    import pandas as pd

    pkg_dir = os.path.dirname(pd.__file__)
    test_dir = os.path.join(pkg_dir, "tests")

    # https://stackoverflow.com/questions/17407119/python-inspect-stack-is-slow
    frame: FrameType | None = inspect.currentframe()
    try:
        n = 0
        while frame:
            filename = inspect.getfile(frame)
            if filename.startswith(pkg_dir) and not filename.startswith(test_dir):
                frame = frame.f_back
                n += 1
            else:
                break
    finally:
        # See note in
        # https://docs.python.org/3/library/inspect.html#inspect.Traceback
        del frame
    return n


def _param_doc(param_type, param_text):
    return f"""{param_type}\n\t{param_text}"""


PARAMS_DOCSTRING = {
    "wsi": _param_doc(
        param_type=":class:`WSIData <wsidata.WSIData>`",
        param_text="The WSIData object to work on.",
    ),
    "key_added": _param_doc(
        param_type="str, default: '{key_added}'",
        param_text="The key to save the result in the WSIData object.",
    ),
}


def _doc(obj=None, *, key_added: str = None):
    """
    A decorator to inject docstring to an object by replacing the placeholder in docstring by looking up a dict.
    """

    def decorator(obj):
        if obj.__doc__ is not None:
            if key_added is not None:
                PARAMS_DOCSTRING["key_added"] = PARAMS_DOCSTRING["key_added"].format(
                    key_added=key_added
                )
            obj.__doc__ = obj.__doc__.format(**PARAMS_DOCSTRING)

        @wraps(obj)
        def wrapper(*args, **kwargs):
            return obj(*args, **kwargs)

        return wrapper

    if obj is None:
        return decorator
    else:
        return decorator(obj)
