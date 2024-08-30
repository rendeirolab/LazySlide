import fsspec
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    DownloadColumn,
    TransferSpeedColumn,
    TimeRemainingColumn,
)


class Downloader:
    def __init__(self, url, dest):
        self.url = url
        self.dest = dest

    def download_pbar(self):
        """Download a single file with progress tracking."""
        progress = Progress(
            TextColumn("[bold blue]{task.fields[filename]}", justify="right"),
            BarColumn(bar_width=20),
            "[progress.percentage]{task.percentage:>3.1f}%",
            "•",
            DownloadColumn(),
            "•",
            TransferSpeedColumn(),
            "•",
            TimeRemainingColumn(),
        )
        with progress:
            with fsspec.open(self.url, "rb") as fsrc:
                total_size = fsrc.size  # Retrieve the total file size
                task_id = progress.add_task(
                    "Downloading test", filename="test", total=total_size
                )
                with fsspec.open(f"{self.dest}/test", "wb") as fdst:
                    progress.start_task(task_id)
                    chunk_size = 1024 * 1024  # 1 MB
                    while chunk := fsrc.read(chunk_size):
                        fdst.write(chunk)
                        progress.advance(task_id, chunk_size)
            progress.refresh()
