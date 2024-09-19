class Key:
    tissue_qc = "qc"
    tile_qc = "qc"
    tissue: str = "tissues"
    tissue_id: str = "tissue_id"
    tiles = "tiles"
    tile_spec: str = "tile_spec"
    annotations: str = "annotations"

    @classmethod
    def tile_graph(cls, name):
        return f"{name}_graph"

    @classmethod
    def feature(cls, name, tile_key=None):
        tile_key = tile_key or cls.tiles
        return f"{name}_{tile_key}"

    @classmethod
    def feature_slide(cls, name, tile_key=None):
        tile_key = tile_key or cls.tiles
        return f"{name}_{tile_key}_slide"
