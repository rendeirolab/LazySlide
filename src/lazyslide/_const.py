class Key:
    tissue: str = "tissues"
    tiles = "tiles"

    @classmethod
    def tile_graph(cls, name):
        return f"{name}_graph"

    @classmethod
    def feature(cls, name, tile_key=None):
        tile_key = tile_key or cls.tiles
        return f"{name}_{tile_key}"
