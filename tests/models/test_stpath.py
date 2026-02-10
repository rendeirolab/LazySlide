import json
import os

import h5py
import numpy as np
import scanpy as sc
import torch
from huggingface_hub import hf_hub_download

from lazyslide.models.tile_prediction.stpath import STFM


def read_assets_from_h5(h5_path, keys=None, skip_attrs=False, skip_assets=False):
    assets = {}
    attrs = {}
    with h5py.File(h5_path, "r") as f:
        if keys is None:
            keys = list(f.keys())
        for key in keys:
            if not skip_assets:
                assets[key] = f[key][:]
            if not skip_attrs:
                if f[key].attrs is not None:
                    attrs[key] = dict(f[key].attrs)
    return assets, attrs


def rescale_coords(coords, new_max=100):
    """
    Rescale coordinates to a specified range while maintaining their shape.

    Parameters:
        coords (torch.Tensor): A tensor of shape (n, 2), where each row contains (x, y) coordinates.
        new_max (float): The maximum value for the new scaled coordinates.

    Returns:
        torch.Tensor: The rescaled coordinates.
    """
    # Find the minimum and maximum values of the coordinates
    min_coords = torch.min(coords, dim=0).values
    max_coords = torch.max(coords, dim=0).values

    # Calculate the range of the coordinates
    coord_range = max_coords - min_coords

    # Rescale the coordinates to the range [0, new_max]
    scaled_coords = (coords - min_coords) / coord_range * new_max

    return scaled_coords


def normalize_coords(coords):
    # coords: [-1, 2]
    coords[:, 0] = coords[:, 0] - coords[:, 0].min()
    coords[:, 1] = coords[:, 1] - coords[:, 1].min()
    return rescale_coords(coords)


# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_path = hf_hub_download(
    "tlhuang/STPath",
    "stfm.pth",
)

stfm = STFM()
# Load pre-trained model weights
stfm.load_state_dict(torch.load(model_path, map_location="cpu"), strict=True)

print(f"Model loaded from {model_path}")
stfm.eval()  # Set model to evaluation mode


# prepare exmaple data
sample_id = "INT2"

source_dataroot = "/Users/simon/rep/lbi/reindero/STPath/example_data"
with open(os.path.join(source_dataroot, "var_50genes.json")) as f:
    hvg_list = json.load(f)["genes"]

data_dict, _ = read_assets_from_h5(os.path.join(source_dataroot, f"{sample_id}.h5"))

# rescale coordinates
coords = data_dict["coords"]
coords = torch.from_numpy(coords).to(device)
coords = normalize_coords(coords)

# embeddings have been derived with Gigapath
embeddings = data_dict["embeddings"]
embeddings = torch.from_numpy(embeddings).to(device)
barcodes = data_dict["barcodes"].flatten().astype(str).tolist()
adata = sc.read_h5ad(os.path.join(source_dataroot, f"{sample_id}.h5ad"))[barcodes, :]

# get mask for gene expression. Since we do not have GE mask all
n_spots = embeddings.shape[0]
mask_token = stfm.tokenizer.ge_tokenizer.mask_token.float()
masked_ge_token = mask_token.repeat(n_spots, 1).to(device)

# get organ token
organ_type = "Kidney"
organ_token = stfm.tokenizer.organ_tokenizer.encode(organ_type, align_first=True)
organ_token = torch.full(
    (embeddings.shape[0],), organ_token, dtype=torch.long, device=device
)

# tech token
tech_type = "Visium"
tech_token = stfm.tokenizer.tech_tokenizer.encode(tech_type, align_first=True)
tech_token = torch.full(
    (embeddings.shape[0],), tech_token, dtype=torch.long, device=device
)


batch_idx = torch.zeros(embeddings.shape[0], dtype=torch.long).to(device)

args = (embeddings, coords, masked_ge_token, batch_idx, tech_token, organ_token)


out = stfm(embeddings, coords, masked_ge_token, batch_idx, tech_token, organ_token)
out = out[:, 2:]  # remove the pad and mask tokens

fout = "/Users/simon/Desktop/predictions_reimpl.npz"
with open(fout, "wb") as f:
    np.savez(f, out.detach().numpy())
    print(f"Wrote output of reimplemented model to {fout}.")
print(f"Shape of output: {out.shape}")


def find_jit_culprit(module, name="root"):
    try:
        # Try to script and save the current module
        scripted = torch.jit.script(module)
        scripted.save(f"temp_{name}.pt")
    except RuntimeError as e:
        if "strides() called on an undefined Tensor" in str(e):
            print(f"CULPRIT FOUND: {name} ({type(module).__name__})")
            # Recurse into children to find the specific leaf node
            for child_name, child_module in module.named_children():
                find_jit_culprit(child_module, f"{name}.{child_name}")
        else:
            print(f"Module {name} failed with a different error: {e}")
    except Exception as e:
        print(f"Module {name} failed: {e}")


# Run it on your model
# find_jit_culprit(stfm)


# 1. Convert the module to TorchScript via scripting
scripted_model = torch.jit.script(stfm, example_inputs=[args])

# 2. Save the scripted model (includes weights)
scripted_model.save("stpath_scripted.pt")

# 3. Later, load it as requested
loaded_jit_model = torch.jit.load("stpath_scripted.pt")
