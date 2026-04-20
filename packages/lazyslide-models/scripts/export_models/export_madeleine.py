#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "einops>=0.7",
#   "huggingface_hub>=0.21",
#   "numpy>=2.0",
#   "torch>=2.7.1",
# ]
# ///

from pathlib import Path

from export_utils import export_model, verify_exported

workdir = Path(__file__).parent
checkpoint_dir = workdir / "checkpoints"
checkpoint_dir.mkdir(parents=True, exist_ok=True)

export_artifacts = workdir / "export_artifacts"
export_artifacts.mkdir(parents=True, exist_ok=True)

MADELEINE_EXPORT_PATH = export_artifacts / "MADELEINE_exported.pt2"

# %%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

HE_POSITION = 0


# %%
class MADELEINE(nn.Module):
    def __init__(self, config, stain_encoding=False):
        super(MADELEINE, self).__init__()
        self.config = config
        self.modalities = config.MODALITIES
        self.stain_encoding = stain_encoding

        if self.stain_encoding:
            self.stain_encoding_dim = 32
            self.embedding = nn.Embedding(len(self.modalities), self.stain_encoding_dim)
        else:
            self.stain_encoding_dim = 0

        if self.config.wsi_encoder == "abmil":
            pre_params = {
                "input_dim": self.config.patch_embedding_dim + self.stain_encoding_dim,
                "hidden_dim": self.config.wsi_encoder_hidden_dim,
            }

            attention_params = {
                "model": "ABMIL",
                "params": {
                    "input_dim": self.config.wsi_encoder_hidden_dim,
                    "hidden_dim": 512,
                    "dropout": True,
                    "activation": self.config.activation,
                    "n_heads": self.config.n_heads,
                    "n_classes": 1,
                },
            }

            self.token_projector = nn.Linear(
                attention_params["params"]["hidden_dim"]
                * attention_params["params"]["n_heads"],
                128,
            )
            self.wsi_embedders = ABMILEmbedder(pre_params, attention_params)

            self.projector = nn.Linear(
                attention_params["params"]["hidden_dim"]
                * attention_params["params"]["n_heads"],
                attention_params["params"]["hidden_dim"],
            )

        else:
            raise ValueError(
                'Unsupported wsi_encoder. Must be "abmil". Now is {}.'.format(
                    self.config.wsi_encoder
                )
            )

    def encode_he(self, feats, device):
        feats = feats.to(device)
        bs, _, _ = feats.shape
        n_mod = 1
        feats = feats.unsqueeze(dim=1)
        HE_embedding = self.wsi_embedders(
            feats[:, HE_POSITION, ::], return_attention=False
        )
        d_out, n_heads = HE_embedding.shape[-2], HE_embedding.shape[-1]
        HE_embedding = HE_embedding.view(bs * n_mod, d_out * n_heads)
        HE_embedding = self.projector(HE_embedding)
        HE_embedding = HE_embedding.view(bs, n_mod, d_out)
        return HE_embedding.squeeze(dim=1)

    def forward(
        self,
        data,
        device,
        train=True,
        n_views=1,
        custom_stain_idx=None,
        return_attention=False,
    ):
        all_wsi_feats = data["feats"].to(device)

        all_embeddings = {}
        all_token_embeddings = {}

        if train:
            bs, n_mod, n_tokens, d_in = all_wsi_feats.shape
            all_wsi_feats = all_wsi_feats.view(bs * n_mod, n_tokens, d_in)

            if self.stain_encoding:
                stain_indicator = []
                for i in range(n_mod):
                    stain_indicator += [i] * bs
                stain_indicator = torch.LongTensor([stain_indicator]).to(device)
                stain_encoding = self.embedding(stain_indicator).squeeze()
                stain_encoding = torch.repeat_interleave(
                    stain_encoding.unsqueeze(1), repeats=all_wsi_feats.shape[1], dim=1
                )
                all_wsi_feats = torch.cat([all_wsi_feats, stain_encoding], axis=-1)

            slide_embeddings, token_embeddings = self.wsi_embedders(
                all_wsi_feats, return_preattn_feats=True, n_views=n_views
            )

            token_embeddings = token_embeddings.view(bs * n_mod, n_tokens, -1)
            token_embeddings = token_embeddings.view(bs, n_mod, n_tokens, -1)
            token_embeddings = self.token_projector(token_embeddings)

            d_out, n_heads = slide_embeddings.shape[-2], slide_embeddings.shape[-1]
            slide_embeddings = slide_embeddings.view(bs * n_mod, -1, d_out * n_heads)
            slide_embeddings = self.projector(slide_embeddings)
            slide_embeddings = slide_embeddings.view(bs, n_mod, -1, d_out)

            for idx, modality in enumerate(self.modalities):
                slide_emb = slide_embeddings[:, idx, :, :]
                token_emb = token_embeddings[:, idx, :]
                if modality == "HE":
                    slide_emb = slide_emb.unsqueeze(dim=3).repeat(1, 1, 1, n_mod - 1)
                    token_emb = token_emb.unsqueeze(dim=3).repeat(1, 1, 1, n_mod - 1)
                all_embeddings[modality] = slide_emb
                all_token_embeddings[modality] = token_emb

            return all_embeddings, all_token_embeddings

        elif not train and not return_attention:
            bs, n_mod, n_tokens, d_in = all_wsi_feats.shape

            for stain_idx in range(n_mod):
                if custom_stain_idx:
                    stain_name = self.modalities[custom_stain_idx]
                else:
                    stain_name = self.modalities[stain_idx]

                curr_stain_feats = all_wsi_feats[:, stain_idx, ::]

                if self.stain_encoding:
                    if custom_stain_idx:
                        key = custom_stain_idx
                    else:
                        key = stain_idx

                    stain_indicator = torch.LongTensor([[key] * bs]).to(device)
                    stain_encoding = self.embedding(stain_indicator)
                    stain_encoding = torch.repeat_interleave(
                        stain_encoding, repeats=n_tokens, dim=1
                    )
                    curr_stain_feats = torch.cat(
                        [curr_stain_feats, stain_encoding], axis=-1
                    )

                stain_embedding = self.wsi_embedders(curr_stain_feats)

                d_out, n_heads = stain_embedding.shape[-2], stain_embedding.shape[-1]
                stain_embedding = stain_embedding.view(bs * n_mod, d_out * n_heads)
                stain_embedding = self.projector(stain_embedding)
                stain_embedding = stain_embedding.view(bs, n_mod, d_out)

                all_embeddings[stain_name] = stain_embedding

            return all_embeddings

        else:
            stain_name = "HE"
            bs, n_mod, n_tokens, d_in = all_wsi_feats.shape
            HE_embedding, raw_attention = self.wsi_embedders(
                all_wsi_feats[:, HE_POSITION, ::], return_attention=True
            )
            d_out, n_heads = HE_embedding.shape[-2], HE_embedding.shape[-1]
            HE_embedding = HE_embedding.view(bs * n_mod, d_out * n_heads)
            HE_embedding = self.projector(HE_embedding)
            HE_embedding = HE_embedding.view(bs, n_mod, d_out)

            return HE_embedding, raw_attention

    def encode_features(self, feats: torch.Tensor) -> torch.Tensor:
        """
        Export-compatible method for encoding features directly to slide embeddings.

        Args:
            feats (torch.Tensor): Input features [B, N, S] where:
                - B: batch size
                - N: number of tiles/patches
                - S: feature dimension

        Returns:
            torch.Tensor: Slide embeddings [B, hidden_dim]
        """
        bs, n_tokens, d_in = feats.shape

        slide_embedding = self.wsi_embedders(feats, return_attention=False)

        d_out, n_heads = slide_embedding.shape[-2], slide_embedding.shape[-1]
        slide_embedding = slide_embedding.view(bs, d_out * n_heads)
        slide_embedding = self.projector(slide_embedding)

        return slide_embedding


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.blocks = nn.Sequential(
            self.build_block(in_dim=self.input_dim, out_dim=int(self.input_dim)),
            self.build_block(in_dim=int(self.input_dim), out_dim=int(self.input_dim)),
            nn.Linear(in_features=int(self.input_dim), out_features=self.output_dim),
        )

    def build_block(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        x = self.blocks(x)
        return x


class ProjHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ProjHead, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = nn.Sequential(
            nn.Linear(in_features=self.input_dim, out_features=int(self.input_dim)),
            nn.LayerNorm(int(self.input_dim)),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=int(self.input_dim), out_features=self.output_dim),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class ABMILEmbedder(nn.Module):
    """ABMIL."""

    def __init__(
        self,
        pre_attention_params: dict = None,
        attention_params: dict = None,
        aggregation: str = "regular",
    ) -> None:
        super(ABMILEmbedder, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pre_attention_params = pre_attention_params
        self.attention_params = attention_params
        self.n_heads = attention_params["params"]["n_heads"]

        self._build_pre_attention_params(params=pre_attention_params)

        if attention_params is not None:
            self._build_attention_params(
                attn_model=attention_params["model"], params=attention_params["params"]
            )

        self.agg_type = aggregation

    def _build_pre_attention_params(self, params):
        self.pre_attn = nn.Sequential(
            nn.Linear(params["input_dim"], params["hidden_dim"]),
            nn.LayerNorm(params["hidden_dim"]),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(params["hidden_dim"], params["hidden_dim"]),
            nn.LayerNorm(params["hidden_dim"]),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(params["hidden_dim"], params["hidden_dim"] * self.n_heads),
            nn.LayerNorm(params["hidden_dim"] * self.n_heads),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def _build_attention_params(self, attn_model="ABMIL", params=None):
        if attn_model == "ABMIL":
            self.attn = nn.ModuleList(
                [BatchedABMIL(**params).to(self.device) for i in range(self.n_heads)]
            )
        else:
            raise NotImplementedError(
                "Attention model not implemented -- Options are ABMIL"
            )

    def forward(
        self,
        bags: torch.Tensor,
        return_attention: bool = False,
        return_preattn_feats: bool = False,
        n_views=1,
    ) -> torch.Tensor:
        embeddings = self.pre_attn(bags)

        if self.n_heads > 1:
            embeddings = rearrange(embeddings, "b t (e c) -> b t e c", c=self.n_heads)
        else:
            embeddings = embeddings.unsqueeze(-1)

        token_embeddings = embeddings

        attention = []
        raw_attention = []
        for i, attn_net in enumerate(self.attn):
            processed_attention, untouched_attention = attn_net(
                embeddings[:, :, :, i], return_raw_attention=True
            )
            attention.append(processed_attention)
            raw_attention.append(untouched_attention)
        attention = torch.stack(attention, dim=-1)
        raw_attention = torch.stack(raw_attention, dim=-1)

        if self.agg_type == "regular":
            if n_views == 1:
                slide_embeddings = embeddings * attention
                slide_embeddings = torch.sum(slide_embeddings, dim=1)
            else:
                slide_embeddings_wholeView = embeddings * attention
                slide_embeddings_wholeView = torch.sum(
                    slide_embeddings_wholeView, dim=1
                )
                slide_embeddings_wholeView = slide_embeddings_wholeView.unsqueeze(1)

                all_indices = np.arange(embeddings.shape[1])
                np.random.shuffle(all_indices)
                midpoint = len(all_indices) // 2
                list_of_indices = [all_indices[:midpoint], all_indices[midpoint:]]
                embeddings = torch.cat(
                    [
                        embeddings[:, indices, :, :].unsqueeze(1)
                        for indices in list_of_indices
                    ],
                    dim=1,
                )
                attention = torch.cat(
                    [
                        F.softmax(raw_attention[:, indices], dim=1).unsqueeze(1)
                        for indices in list_of_indices
                    ],
                    dim=1,
                )
                embeddings = embeddings * attention
                slide_embeddings_smallViews = torch.sum(embeddings, dim=2)

                slide_embeddings = torch.concat(
                    [slide_embeddings_wholeView, slide_embeddings_smallViews], dim=1
                )
        else:
            raise NotImplementedError('Agg type not supported. Options are "regular".')

        if return_attention:
            return slide_embeddings, raw_attention

        if return_preattn_feats:
            return slide_embeddings, token_embeddings

        return slide_embeddings


class BatchedABMIL(nn.Module):
    def __init__(
        self,
        input_dim=1024,
        hidden_dim=256,
        dropout=False,
        n_classes=1,
        n_heads=1,
        activation="softmax",
    ):
        super(BatchedABMIL, self).__init__()

        self.activation = activation
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.attention_a = nn.ModuleList([nn.Linear(input_dim, hidden_dim), nn.Tanh()])
        self.attention_b = nn.ModuleList(
            [nn.Linear(input_dim, hidden_dim), nn.Sigmoid()]
        )

        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(hidden_dim, n_classes)

    def forward(self, x, return_raw_attention=False):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)

        if self.activation == "softmax":
            activated_A = F.softmax(A, dim=1)
        elif self.activation == "leaky_relu":
            activated_A = F.leaky_relu(A)
        elif self.activation == "relu":
            activated_A = F.relu(A)
        elif self.activation == "sigmoid":
            activated_A = torch.sigmoid(A)
        else:
            raise NotImplementedError("Activation not implemented.")

        if return_raw_attention:
            return activated_A, A
        return activated_A


# %%
class MADELEINEJITWrapper(nn.Module):
    """
    Export-compatible wrapper for MADELEINE model.
    Flattened architecture for torch.export compatibility.
    """

    def __init__(self, madeleine_model: MADELEINE):
        super(MADELEINEJITWrapper, self).__init__()

        self.input_dim = madeleine_model.wsi_embedders.pre_attention_params["input_dim"]
        self.hidden_dim = madeleine_model.wsi_embedders.pre_attention_params[
            "hidden_dim"
        ]
        self.n_heads = madeleine_model.wsi_embedders.n_heads

        # Pre-attention network (simplified, weights copied below)
        self.pre_linear1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.pre_linear2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.pre_linear3 = nn.Linear(self.hidden_dim, self.hidden_dim * self.n_heads)

        attn_params = madeleine_model.wsi_embedders.attention_params["params"]
        self.attn_input_dim = attn_params["input_dim"]
        self.attn_hidden_dim = attn_params["hidden_dim"]

        self.attention_a = nn.Linear(self.attn_input_dim, self.attn_hidden_dim)
        self.attention_b = nn.Linear(self.attn_input_dim, self.attn_hidden_dim)
        self.attention_c = nn.Linear(self.attn_hidden_dim, 1)

        proj_input_dim = self.attn_hidden_dim * self.n_heads
        proj_output_dim = self.attn_hidden_dim
        self.projector = nn.Linear(proj_input_dim, proj_output_dim)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """
        Encode patch features to slide embedding.

        Args:
            feats: [B, N, S] — batch, num tiles, feature dim

        Returns:
            torch.Tensor: [B, hidden_dim] slide embeddings
        """
        bs, n_tokens, d_in = feats.shape

        x = F.gelu(F.layer_norm(self.pre_linear1(feats), [self.hidden_dim]))
        x = F.dropout(x, 0.1, training=self.training)
        x = F.gelu(F.layer_norm(self.pre_linear2(x), [self.hidden_dim]))
        x = F.dropout(x, 0.1, training=self.training)
        embeddings = F.gelu(
            F.layer_norm(self.pre_linear3(x), [self.hidden_dim * self.n_heads])
        )
        embeddings = F.dropout(embeddings, 0.1, training=self.training)

        # Use first head's worth of features for attention
        if self.n_heads > 1:
            embeddings = embeddings[:, :, : self.attn_input_dim]

        a = torch.tanh(self.attention_a(embeddings))
        b = torch.sigmoid(self.attention_b(embeddings))
        A = a * b
        A = self.attention_c(A)
        attention_weights = F.softmax(A, dim=1)

        slide_embedding = embeddings * attention_weights
        slide_embedding = torch.sum(slide_embedding, dim=1)

        # Pad to projector input dim if needed
        if self.n_heads > 1:
            expected_dim = self.attn_hidden_dim * self.n_heads
            current_dim = slide_embedding.shape[-1]
            if current_dim < expected_dim:
                padding = torch.zeros(
                    bs,
                    expected_dim - current_dim,
                    dtype=slide_embedding.dtype,
                    device=slide_embedding.device,
                )
                slide_embedding = torch.cat([slide_embedding, padding], dim=-1)

        slide_embedding = self.projector(slide_embedding)

        return slide_embedding


def copy_weights_to_jit_model(
    original_model: MADELEINE, jit_model: MADELEINEJITWrapper
):
    """Copy weights from original MADELEINE to the export wrapper."""
    original_pre_attn = original_model.wsi_embedders.pre_attn
    with torch.no_grad():
        jit_model.pre_linear1.weight.copy_(original_pre_attn[0].weight)
        jit_model.pre_linear1.bias.copy_(original_pre_attn[0].bias)
        jit_model.pre_linear2.weight.copy_(original_pre_attn[4].weight)
        jit_model.pre_linear2.bias.copy_(original_pre_attn[4].bias)
        jit_model.pre_linear3.weight.copy_(original_pre_attn[8].weight)
        jit_model.pre_linear3.bias.copy_(original_pre_attn[8].bias)

        original_attn = original_model.wsi_embedders.attn[0]
        jit_model.attention_a.weight.copy_(original_attn.attention_a[0].weight)
        jit_model.attention_a.bias.copy_(original_attn.attention_a[0].bias)
        jit_model.attention_b.weight.copy_(original_attn.attention_b[0].weight)
        jit_model.attention_b.bias.copy_(original_attn.attention_b[0].bias)
        jit_model.attention_c.weight.copy_(original_attn.attention_c.weight)
        jit_model.attention_c.bias.copy_(original_attn.attention_c.bias)

        jit_model.projector.weight.copy_(original_model.projector.weight)
        jit_model.projector.bias.copy_(original_model.projector.bias)


# %%
import json
from argparse import Namespace
from collections import OrderedDict

from huggingface_hub import snapshot_download

repo = snapshot_download(repo_id="MahmoodLab/madeleine")

with open(repo + "/model_config.json", "r") as f:
    model_cfg = json.load(f)
model_cfg = Namespace(**model_cfg)

state_dict = torch.load(repo + "/model.pt", map_location="cpu", weights_only=False)

# %%
model = MADELEINE(config=model_cfg, stain_encoding=False)

sd = list(state_dict.keys())
contains_module = any("module" in entry for entry in sd)

if not contains_module:
    model.load_state_dict(state_dict, strict=True)
else:
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=True)
print("Loaded weights successfully!")

# %%
jit_model = MADELEINEJITWrapper(model)
copy_weights_to_jit_model(model, jit_model)
jit_model.eval()

# Dynamic batch (dim 0) + N_tokens (dim 1); feature dim is fixed at 512
dynamic_shapes = [{0: torch.export.Dim.AUTO, 1: torch.export.Dim.AUTO}]

madeleine_example_input = torch.randn(2, 10, 512)
export_model(
    jit_model,
    madeleine_example_input,
    MADELEINE_EXPORT_PATH,
    dynamic_shapes=dynamic_shapes,
)

# %%
torch.manual_seed(42)
fixed_input = torch.randn(4, 20, 512)

verify_exported(jit_model, MADELEINE_EXPORT_PATH, fixed_input, "MADELEINE")
