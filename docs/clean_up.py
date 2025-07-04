import shutil
from pathlib import Path

root = Path(__file__).parent  # ./docs

target_folders = [
    root / "build",
    root / "source" / "api" / "_autogen",
    root / "jupyter_execute",
]


if __name__ == "__main__":
    for folder in target_folders:
        if folder.exists():
            shutil.rmtree(folder)
