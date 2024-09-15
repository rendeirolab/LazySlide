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
    from rich.progress import Progress
    from rich.progress import (
        TextColumn,
        BarColumn,
        TaskProgressColumn,
        TimeRemainingColumn,
    )

    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=30),
        TaskProgressColumn(),
        TimeRemainingColumn(compact=True, elapsed_when_finished=True),
        disable=disable,
        console=console,
    )


def chunker(seq, num_workers):
    avg = len(seq) / num_workers
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last) : int(last + avg)])
        last += avg

    return out
