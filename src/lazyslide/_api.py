from contextlib import nullcontext

from ._setting import settings


def default_value(name, value=None):
    if value is None:
        return getattr(settings, name)
    else:
        return value


def autocast(device, amp=None, autocast_dtype=None):
    """Return an AMP context for `device`, or a no-op context when amp is off."""
    import torch

    if not default_value("amp", amp):
        return nullcontext()
    # torch.autocast expects a device *type* string, not a torch.device
    return torch.autocast(
        torch.device(device).type,
        dtype=default_value("autocast_dtype", autocast_dtype),
    )


def maybe_compile(model, compile=None, compile_kws=None):
    """Compile `model` in place when requested.

    No-op for objects without ``try_compile``, e.g. a plain callable or a
    ``torch.load``-ed object supplied by the user.
    """
    if not default_value("compile", compile):
        return model
    try_compile = getattr(model, "try_compile", None)
    if try_compile is not None:
        try_compile(**default_value("compile_kws", compile_kws))
    return model


def loader_kws(device, num_workers=0, prefetch_factor=None):
    """DataLoader kwargs tuned for `device`."""
    import torch

    kws = {
        "num_workers": num_workers,
        # pin_memory only buys anything for CUDA host-to-device copies
        "pin_memory": torch.device(device).type == "cuda",
    }
    if num_workers > 0 and prefetch_factor is not None:
        # torch rejects prefetch_factor when num_workers == 0
        kws["prefetch_factor"] = prefetch_factor
    return kws
