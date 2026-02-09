import json
import os

import h5py
import scanpy as sc
import torch

from lazyslide.models.tile_prediction.stpath import STPath


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


# init model
stpath = STPath()
stfm = stpath.stfm

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
mask_token = stpath.tokenizer.ge_tokenizer.mask_token.float()
masked_ge_token = mask_token.repeat(n_spots, 1).to(device)

# get organ token
organ_type = "Kidney"
organ_token = stpath.tokenizer.organ_tokenizer.encode(organ_type, align_first=True)
organ_token = torch.full(
    (embeddings.shape[0],), organ_token, dtype=torch.long, device=device
)

# tech token
tech_type = "Visium"
tech_token = stpath.tokenizer.tech_tokenizer.encode(tech_type, align_first=True)
tech_token = torch.full(
    (embeddings.shape[0],), tech_token, dtype=torch.long, device=device
)


batch_idx = torch.zeros(embeddings.shape[0], dtype=torch.long).to(device)

# args must be something like:
#   img_tokens: torch.Tensor,
#        coords: torch.Tensor,
#        ge_tokens: torch.Tensor,
#        batch_idx: torch.Tensor,
#        tech_tokens: torch.Tensor | None = None,
#        organ_tokens: torch.Tensor | None = None,
#        return_all=False,

args = (embeddings, coords, masked_ge_token, batch_idx, tech_token, organ_token)


out = stfm(embeddings, coords, masked_ge_token, batch_idx, tech_token, organ_token)

print(out.shape)


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
# with open("/Users/simon/Desktop/scripted_model", "w") as f:
# f.write(scripted_model.code)

# print(scripted_model.get_debug_state())

# Print the graph before saving
print(scripted_model.graph)

# Specifically look for 'strides' or 'UndefinedTensor' in the IR
# You can also use:
print(scripted_model.code)
#
# # 2. Save the scripted model (includes weights)
scripted_model.save("stpath_scripted.pt")
#
# # 3. Later, load it as requested
loaded_jit_model = torch.jit.load("stpath_scripted.pt")

print(loaded_jit_model.code)
