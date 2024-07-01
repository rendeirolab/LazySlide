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


def check_feature_key(wsi, feature_key, tile_key=None):
    """Check if the feature key exists in the wsi data"""
    if tile_key is None:
        if feature_key not in wsi.sdata.tables:
            raise ValueError(f"`{feature_key}` not found in the table.")
        return feature_key
    else:
        if feature_key not in wsi.sdata.tables:
            if f"{tile_key}_{feature_key}" not in wsi.sdata.tables:
                raise ValueError(
                    f"Either `{feature_key}` or "
                    f"`{tile_key}_{feature_key}` not found in the table."
                )
            return f"{tile_key}_{feature_key}"
        return feature_key
