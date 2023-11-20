from typing import Union
from pathlib import Path

from .base import BackendBase


try:
    import pyvips as vips
except Exception as e:
    pass


class VipsBackend(BackendBase):

    def __init__(self,
                 file: Union[Path, str],

                 ):

        self.file = Path(file)
        self.metadata = self.get_metadata()
