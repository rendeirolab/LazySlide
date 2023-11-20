from dataclasses import dataclass, field


class BackendBase:

    def get_patch(self):
        """Get a patch from image with top-left corner"""
        raise NotImplemented

    def get_cell(self):
        """Get a patch from image with center"""
        raise NotImplemented

    def get_metadata(self):
        raise NotImplemented


@dataclass
class WSIMetaData:
    file_id: str
    mpp: field(None)
