# Base class
class Transform:
    """
    Base class for all Transforms.
    Each transform must operate on a Tile.
    """

    def __repr__(self):
        return "Base class for all transforms"

    def apply(self, image):
        """Perform transformation"""
        raise NotImplementedError
