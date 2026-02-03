import torch
from einops import rearrange
from torch import nn

from ..base import TilePredictionModel


# @register(
#     key="pathprofilerqc",
#     task=ModelTask.tile_prediction,
#     license="GPL-3.0",
#     description="Quality assessment of histology images",
#     commercial=False,
#     github_url="https://github.com/MaryamHaghighat/PathProfiler",
#     paper_url="https://doi.org/10.1038/s41598-022-08351-5",
#     bib_key="Haghighat2022-sy",
#     param_size="11.2M",
#     flops="3.63G",
# )
class STPath(TilePredictionModel):
    def __init__(self, model_path=None, token=None):
        from huggingface_hub import hf_hub_download

        self._model_file = hf_hub_download(
            "tlhuang/STPath",
            "stfm.pth",
        )

    def predict(self, image):
        return None


class STFM(nn.Module):
    def __init__(self, config) -> None:
        super(STFM, self).__init__()

        self.backbone = config.backbone


class SpatialTransformer(nn.Module):
    def __init__(self, config):
        super(SpatialTransformer, self).__init__()

        self.blks = nn.ModuleList(
            [
                TransformerBlock(
                    config.d_model,
                    n_heads=config.n_heads,
                    activation=config.act,
                    attn_drop=config.attn_dropout,
                    proj_drop=config.dropout,
                    mlp_ratio=config.mlp_ratio,
                )
                for i in range(config.n_layers)
            ]
        )

    def forward(self, features, coords, batch_idx, **kwargs):
        # apply the same mask to all cells in the same batch
        batch_mask = ~(batch_idx.unsqueeze(0) == batch_idx.unsqueeze(1))

        # forward pass
        features = features.unsqueeze(0)  # [1, N_cells, d_model]
        coords = coords.unsqueeze(0)  # [1, N_cells, 2]
        batch_mask = batch_mask.unsqueeze(0)  # [1, N_cells, N_cells]
        for blk in self.blks:
            features = blk(features, coords, padding_mask=batch_mask)
        return features.squeeze(0)  # [N_cells, N_genes]


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads=1,
        activation="gelu",
        attn_drop=0.0,
        proj_drop=0.0,
        mlp_ratio=4.0,
    ):
        super(TransformerBlock, self).__init__()

        self.attn = Attention(
            d_model=d_model, n_heads=n_heads, proj_drop=proj_drop, attn_drop=attn_drop
        )

        self.mlp = MLPWrapper(
            in_features=d_model,
            hidden_features=int(d_model * mlp_ratio),
            out_features=d_model,
            activation=activation,
            drop=proj_drop,
            norm_layer=nn.LayerNorm,
        )

    def forward(self, token_embs, coords, padding_mask=None):
        context_token_embs = self.attn(token_embs, coords, padding_mask)
        token_embs = token_embs + context_token_embs

        token_embs = token_embs + self.mlp(token_embs)
        return token_embs


class FrameAveraging(nn.Module):
    def __init__(self, dim=3, backward=False):
        super(FrameAveraging, self).__init__()

        self.dim = dim
        self.n_frames = 2**dim
        self.ops = self.create_ops(dim)  # [2^dim, dim]
        self.backward = backward

    def create_ops(self, dim):
        colon = slice(None)
        accum = []
        directions = torch.tensor([-1, 1])

        for ind in range(dim):
            dim_slice = [None] * dim
            dim_slice[ind] = colon
            accum.append(directions[dim_slice])

        accum = torch.broadcast_tensors(*accum)
        operations = torch.stack(accum, dim=-1)
        operations = rearrange(operations, "... d -> (...) d")
        return operations

    def create_frame(self, X, mask=None):
        assert X.shape[-1] == self.dim, (
            f"expected points of dimension {self.dim}, but received {X.shape[-1]}"
        )

        if mask is None:
            mask = torch.ones(*X.shape[:-1], device=X.device).bool()
        mask = mask.unsqueeze(-1)
        center = (X * mask).sum(dim=1) / mask.sum(dim=1)
        X = X - center.unsqueeze(1) * mask  # [B,N,dim]
        X_ = X.masked_fill(~mask, 0.0)

        C = torch.bmm(X_.transpose(1, 2), X_)  # [B,dim,dim] (Cov)
        if not self.backward:
            C = C.detach()

        _, eigenvectors = torch.linalg.eigh(C, UPLO="U")  # [B,dim,dim]
        F_ops = self.ops.unsqueeze(1).unsqueeze(0).to(
            X.device
        ) * eigenvectors.unsqueeze(
            1
        )  # [1,2^dim,1,dim] x [B,1,dim,dim] -> [B,2^dim,dim,dim]
        h = torch.einsum(
            "boij,bpj->bopi", F_ops.transpose(2, 3), X
        )  # transpose is inverse [B,2^dim,N,dim]

        h = h.view(X.size(0) * self.n_frames, X.size(1), self.dim)
        return h, F_ops.detach(), center

    def invert_frame(self, X, mask, F_ops, center):
        X = torch.einsum("boij,bopj->bopi", F_ops, X)
        X = X.mean(dim=1)  # frame averaging
        X = X + center.unsqueeze(1)
        if mask is None:
            return X
        return X * mask.unsqueeze(-1)


class Attention(FrameAveraging):
    def __init__(
        self,
        d_model,
        n_heads=1,
        proj_drop=0.0,
        attn_drop=0.0,
        max_n_tokens=5e3,
    ):
        super(Attention, self).__init__(dim=2)

        self.max_n_tokens = max_n_tokens
        self.d_head, self.n_heads = d_model // n_heads, n_heads
        self.scale = self.d_head**-0.5

        self.layernorm_qkv = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 3),
        )
        self.W_output = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(proj_drop),
        )
        self.attn_dropout = nn.Dropout(attn_drop)
        self.edge_bias = nn.Sequential(
            nn.Linear(self.dim + 1, self.n_heads, bias=False),
        )

    def forward(self, x, coords, pad_mask: torch.Tensor = None):
        B, N, C = x.shape
        q, k, v = self.layernorm_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda x: rearrange(x, "b n (h d) -> b h n d", h=self.n_heads), (q, k, v)
        )

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        """build pairwise representation with FA"""
        radial_coords = coords.unsqueeze(dim=2) - coords.unsqueeze(
            dim=1
        )  # [B, N, N, 2]
        radial_coord_norm = radial_coords.norm(dim=-1).reshape(
            B * N, N, 1
        )  # [B*N, N, 1]

        radial_coords = rearrange(radial_coords, "b n m d -> (b n) m d")
        neighbor_masks = (
            ~rearrange(pad_mask, "b n m -> (b n) m") if pad_mask is not None else None
        )
        frame_feats, _, _ = self.create_frame(
            radial_coords, neighbor_masks
        )  # [B*N*4, N, 2]
        frame_feats = frame_feats.view(
            B * N, self.n_frames, N, -1
        )  # [N, 4, N, d_model]

        radial_coord_norm = radial_coord_norm.unsqueeze(dim=1).expand(
            B * N, self.n_frames, N, -1
        )

        # for efficiency
        spatial_bias = self.edge_bias(
            torch.cat([frame_feats, radial_coord_norm], dim=-1)
        ).mean(dim=1)  # [B * N, N, n_heads]
        spatial_bias = rearrange(spatial_bias, "(b n) m h -> b h n m", b=B, n=N)

        """add spatial bias"""
        attn = attn + spatial_bias
        if pad_mask is not None:
            pad_mask = pad_mask.unsqueeze(1)
            attn.masked_fill_(pad_mask, -1e9)
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        return self.W_output(x)


def get_activation(activation="gelu"):
    return {
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "relu": nn.ReLU,
    }[activation]


class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
        drop_last=True,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = (
            norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        )
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop) if drop_last else nn.Identity()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SwiGLUMLP(nn.Module):
    """MLP w/ GLU style gating
    See: https://arxiv.org/abs/1612.08083, https://arxiv.org/abs/2002.05202
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        norm_layer=None,
        bias=True,
        drop=0.0,
        drop_last=True,
    ):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        assert hidden_features % 2 == 0

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = nn.SiLU()
        self.drop1 = nn.Dropout(drop)
        self.norm = (
            norm_layer(hidden_features // 2)
            if norm_layer is not None
            else nn.Identity()
        )
        self.fc2 = nn.Linear(hidden_features // 2, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop) if drop_last else nn.Identity()

    def init_weights(self):
        # override init of fc1 w/ gate portion set to weight near zero, bias=1
        fc1_mid = self.fc1.bias.shape[0] // 2
        nn.init.ones_(self.fc1.bias[fc1_mid:])
        nn.init.normal_(self.fc1.weight[fc1_mid:], std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x1, x2 = x.chunk(2, dim=-1)
        x = self.act(x1) * x2
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


def MLPWrapper(
    in_features,
    hidden_features,
    out_features,
    activation="gelu",
    norm_layer=None,
    bias=True,
    drop=0.0,
    drop_last=True,
):
    if activation == "swiglu":
        return SwiGLUMLP(
            in_features,
            hidden_features,
            out_features,
            norm_layer,
            bias,
            drop,
            drop_last,
        )
    else:
        return MLP(
            in_features,
            hidden_features,
            out_features,
            get_activation(activation),
            norm_layer,
            bias,
            drop,
            drop_last,
        )
