import json
from typing import Protocol, runtime_checkable

import numpy as np
import scanpy as sc
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from ..base import TilePredictionModel


@runtime_checkable
class TokenizerBase(Protocol):
    def encode(self, *args, **kwargs): ...

    def decode(self, *args, **kwargs): ...

    @property
    def mask_token(self): ...

    @property
    def mask_token_id(self) -> int: ...

    @property
    def pad_token(self): ...

    @property
    def pad_token_id(self) -> int: ...


class GeneExpTokenizer(TokenizerBase):
    def __init__(
        self,
        symbol2gene_path,
    ):
        self.symbol2gene = json.load(open(symbol2gene_path, "r"))
        gene2id = list(set(self.symbol2gene.values()))
        gene2id.sort()  # make sure the order is fixed
        self.gene2id = {gene_id: i for i, gene_id in enumerate(gene2id)}

        # add 2 for pad and mask token
        self.gene2id = {gene: token_id + 2 for gene, token_id in self.gene2id.items()}

    def get_available_symbols(self):
        return [
            g
            for g in list(self.symbol2gene.keys())
            if self.symbol2gene[g] in self.gene2id
        ]

    def get_available_genes(self):
        # rank the genes by their id
        return sorted(self.gene2id.keys(), key=lambda x: self.gene2id[x])

    def convert_gene_exp_to_one_hot_tensor(self, n_genes, gene_exp, gene_ids):
        # gene_exp: [N_cells, N_genes], gene_ids: [N_genes]
        gene_ids = gene_ids.clamp(min=0, max=n_genes)[None, ...].expand(
            gene_exp.shape[0], -1
        )
        gene_exp_onehot = torch.zeros(gene_exp.size(0), n_genes, device=gene_exp.device)
        gene_exp_onehot.scatter_(dim=1, index=gene_ids, src=gene_exp)
        return gene_exp_onehot

    def symbol2id(self, symbol_list, return_valid_positions=False):
        # if return_valid_positions is True, return the positions of the valid symbols in the input list
        res = [
            self.gene2id[self.symbol2gene[symbol]]
            for symbol in symbol_list
            if symbol in self.symbol2gene
        ]
        if len(res) != len(symbol_list):
            print(
                f"Warning: {len(symbol_list) - len(res)} symbols are not in the tokenizer."
            )

        if return_valid_positions:
            valid_positions = [
                i for i, symbol in enumerate(symbol_list) if symbol in self.symbol2gene
            ]
            return res, valid_positions
        else:
            return res

    def shift_token_id(self, n_shift):
        self.gene2id = {
            gene: token_id + n_shift for gene, token_id in self.gene2id.items()
        }

    def subset_gene_ids(self, symbol_list):
        gene_ids = [self.symbol2gene[s] for s in symbol_list if s in self.symbol2gene]
        # only keep the valid gene ids in self.gene2ids
        self.gene2id = {gene_id: i for i, gene_id in enumerate(gene_ids)}

    def get_hvg(self, adata, n_top_genes=100):
        if "raw_hvg_names" in adata.uns:
            hvg_names = adata.uns["raw_hvg_names"][:n_top_genes]
        else:
            all_gene_names = set(self.get_available_symbols())
            available_genes = [
                g for g in adata.var_names.tolist() if g in all_gene_names
            ]
            adata = adata.copy()[:, available_genes]

            sc.pp.filter_genes(adata, min_cells=np.ceil(0.1 * len(adata.obs)))
            sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
            hvg_names = adata.var_names[adata.var["highly_variable"]][
                :n_top_genes
            ].tolist()

        hvg_names = [
            g
            for g in hvg_names
            if g in self.symbol2gene and self.symbol2gene[g] is not None
        ]
        hvg_gene_ids = np.array(
            [
                self.gene2id[self.symbol2gene[g]]
                for g in hvg_names
                if self.symbol2gene[g] in self.gene2id
            ]
        )
        return hvg_names, torch.from_numpy(hvg_gene_ids).long()

    def encode(self, adata, return_sparse=False):
        gene_list = [
            g
            for g in adata.var_names.tolist()
            if g in self.symbol2gene
            and self.symbol2gene[g] is not None
            and self.symbol2gene[g] in self.gene2id
        ]
        adata = adata[:, gene_list]

        if not isinstance(adata.X, np.ndarray):
            # row = create_row_index_tensor(adata.X)
            counts_per_row = np.diff(adata.X.indptr)
            row = np.repeat(
                np.arange(adata.X.shape[0]), counts_per_row
            )  # create row indices for the sparse matrix
            col = adata.X.indices
            act_gene_exp = adata.X.data  # nonzero values
        else:
            row, col = np.nonzero(adata.X)
            act_gene_exp = adata.X[row, col]

        obs_gene_ids = np.array(
            [self.gene2id[self.symbol2gene[g]] for g in adata.var_names.tolist()]
        )  # TODO: there might be some genes sharing the same id so some expression values will be missing
        col = obs_gene_ids[col]
        obs_gene_ids = torch.from_numpy(obs_gene_ids).long()
        act_gene_exp = torch.from_numpy(act_gene_exp).float()

        indices = torch.stack(
            [torch.from_numpy(row), torch.from_numpy(col)], dim=0
        ).long()

        if return_sparse:
            try:
                from torch_geometric.utils import coalesce
            except ImportError:
                print("torch_geometric is not installed.")
            # remove duplicate indices
            indices, act_gene_exp = coalesce(
                edge_index=indices,
                edge_attr=act_gene_exp,
                reduce="max",  # pick the maximum value if there are multiple values for the same edge
            )
            gene_exp = torch.sparse_coo_tensor(
                indices, act_gene_exp, size=(adata.shape[0], self.n_tokens)
            )
        else:
            gene_exp = adata.to_df().values
            gene_exp = torch.from_numpy(gene_exp).float()
            gene_exp = self.convert_gene_exp_to_one_hot_tensor(
                self.n_tokens, gene_exp, obs_gene_ids
            )

        # gene_exp: [n_cells, n_genes], obs_gene_ids: [-1]
        return gene_exp, obs_gene_ids

    @property
    def n_tokens(self):
        return max(self.gene2id.values()) + 1

    @property
    def mask_token(self):
        return F.one_hot(torch.tensor(1), num_classes=self.n_tokens).float()

    @property
    def mask_token_id(self) -> int:
        return 1

    @property
    def pad_token(self):
        return F.one_hot(torch.tensor(0), num_classes=self.n_tokens).float()

    @property
    def pad_token_id(self) -> int:
        return 0


# To accomodate the image features,
# the mask token and pad token are defined as one-hot vectors
# with the same dimension as the image features
class ImageTokenizer(TokenizerBase):
    def __init__(
        self,
        feature_dim: int,
    ):
        super().__init__()

        self.feature_dim = feature_dim  # dimension of image features, e.g., 1024

    @property
    def n_tokens(self):
        return 2

    @property
    def mask_token(self):
        return F.one_hot(torch.tensor(1), num_classes=self.feature_dim).float()

    @property
    def mask_token_id(self) -> int:
        return 1

    @property
    def pad_token(self):
        return F.one_hot(torch.tensor(0), num_classes=self.feature_dim).float()

    @property
    def pad_token_id(self) -> int:
        return 0


# technology
tech_align_mapping = {
    "ST": "Spatial Transcriptomics",
    "Visium": "Visium",
    "Xenium": "Xenium",
    "VisiumHD": "Visium HD",
    "Visium HD": "Visium HD",
    "Spatial Transcriptomics": "Spatial Transcriptomics",
}
tech_voc = ["<pad>", "Spatial Transcriptomics", "Visium", "Xenium", "Visium HD"]


# species
specie_align_mapping = {
    "Mus musculus": "Mus musculus",
    "Homo sapiens": "Homo sapiens",
    "human": "Homo sapiens",
    "mouse": "Mus musculus",
    "plant": "others",
    "rattus norvegicus": "others",
    "human & mouse": "others",
    "pig": "others",
    "fish": "others",
    "frog": "others",
    "ambystoma mexicanum": "others",
}
specie_voc = ["<pad>", "<mask>", "<unk>", "Mus musculus", "Homo sapiens", "others"]


organ_align_mapping = {
    "Spinal cord": "Spinal cord",
    "Brain": "Brain",
    "Breast": "Breast",
    "Bowel": "Bowel",
    "Skin": "Skin",
    "Heart": "Heart",
    "Kidney": "Kidney",
    "Prostate": "Prostate",
    "Lung": "Lung",
    "Liver": "Liver",
    "Uterus": "Uterus",
    "Bone": "Bone",
    "Muscle": "Muscle",
    "Eye": "Eye",
    "Pancreas": "Pancreas",
    "breast": "Breast",
    "brain": "Brain",
    "kidney": "Kidney",
    "heart": "Heart",
    "skin": "Skin",
    "liver": "Liver",
    "pancreas": "Pancreas",
    "mouth": "Mouth",
    "ovary": "Ovary",
    "prostate": "Prostate",
    "glioma": "Glioma",
    "glioblastoma": "Glioblastoma",
    "stomach": "Stomach",
    "colon": "Colon",
    "lung": "Lung",
    "muscle": "Muscle",
    "Bladder": "Others",
    "Lymphoid": "Others",
    "Cervix": "Others",
    "Lymph node": "Others",
    "Ovary": "Others",
    "Embryo": "Others",
    "Lung/Brain": "Others",
    "Kidney/Brain": "Others",
    "Placenta": "Others",
    "Whole organism": "Others",
    "thymus": "Others",
    "joint": "Others",
    "undifferentiated pleomorphic sarcoma": "Others",
    "largeintestine": "Others",
    "lacrimal gland": "Others",
    "leiomyosarcoma": "Others",
    "endometrium": "Others",
    "brain+kidney": "Others",
    "cerebellum": "Others",
    "cervix": "Others",
    "colorectal": "Others",
    "lymphnode": "Others",
}
organ_voc = [
    "<pad>",
    "<mask>",
    "<unk>",
    "Spinal cord",
    "Brain",
    "Breast",
    "Bowel",
    "Skin",
    "Heart",
    "Kidney",
    "Prostate",
    "Lung",
    "Liver",
    "Uterus",
    "Bone",
    "Muscle",
    "Eye",
    "Pancreas",
    "Mouth",
    "Ovary",
    "Glioma",
    "Glioblastoma",
    "Stomach",
    "Colon",
    "Others",
]

# annotation
cancer_annotation_align_mapping = {
    "invasive": "tumor",
    "invasive cancer": "tumor",
    "tumor": "tumor",
    "surrounding tumor": "tumor",
    "immune infiltrate": "tumor",
    "cancer in situ": "tumor",
    "tumor stroma with inflammation": "tumor",
    "tumor cells": "tumor",
    "tumour cells": "tumor",
    "tumor stroma fibrous": "tumor",
    "tumor stroma": "tumor",
    "fibrosis": "tumor",
    "high tils stroma": "tumor",
    "in situ carcinoma*": "tumor",
    "in situ carcinoma": "tumor",
    "tumor cells ?": "tumor",
    "hyperplasia": "tumor",
    "tumor cells - spindle cells": "tumor",
    "tumour stroma": "tumor",
    "necrosis": "tumor",
    "tumor_edge_5": "tumor",
    "idc_4": "tumor",
    "idc_3": "tumor",
    "idc_2": "tumor",
    "tumor_edge_3": "tumor",
    "idc_5": "tumor",
    "dcis/lcis_4": "tumor",
    "idc_7": "tumor",
    "tumor_edge_1": "tumor",
    "dcis/lcis_1": "tumor",
    "dcis/lcis_2": "tumor",
    "tumor_edge_4": "tumor",
    "idc_1": "tumor",
    "benign": "tumor",
    "gg4 cribriform": "tumor",
    "gg2": "tumor",
    "chronic inflammation": "tumor",
    "gg1": "tumor",
    "transition_state": "tumor",
    "benign*": "tumor",
    "gg4": "tumor",
    "pin": "tumor",
    "inflammation": "tumor",
    "dcis/lcis_3": "tumor",
    "tumor_edge_2": "tumor",
    "dcis/lcis_5": "tumor",
    "idc_6": "tumor",
    "tumor_edge_6": "tumor",
    "tls": "tumor",
    "t_agg": "tumor",
    "healthy": "healthy",
    "non tumor": "healthy",
    "normal": "healthy",
    "breast glands": "healthy",
    "connective tissue": "healthy",
    "adipose tissue": "healthy",
    "artifacts": "healthy",
    "normal epithelium": "healthy",
    "lymphocytes": "healthy",
    "healthy_2": "healthy",
    "vascular": "healthy",
    "peripheral nerve": "healthy",
    "lymphoid stroma": "healthy",
    "fibrous stroma": "healthy",
    "fibrosis (peritumoral)": "healthy",
    "artefacts": "healthy",
    "endothelial": "healthy",
    "healthy_1": "healthy",
    "nerve": "healthy",
    "fat": "healthy",
    "stroma": "healthy",
    "exclude": "healthy",
    "vessel": "healthy",
    "no_tls": "healthy",
}

cancer_annotation_voc = ["<pad>", "<mask>", "<unk>", "healthy", "tumor"]

domain_annotation_align_mapping = {
    "l1": "l1",
    "l2": "l2",
    "l3": "l3",
    "l4": "l4",
    "l5": "l5",
    "l6": "l6",
    "vm": "vm",
}

domain_annotation_voc = [
    "<pad>",
    "<mask>",
    "<unk>",
    "l1",
    "l2",
    "l3",
    "l4",
    "l5",
    "l6",
    "vm",
]


class IDTokenizer(TokenizerBase):
    def __init__(
        self,
        id_type="tech",
    ):
        if id_type == "tech":
            token_aligner = tech_align_mapping
            all_tokens = tech_voc
        elif id_type == "specie":
            token_aligner = specie_align_mapping
            all_tokens = specie_voc
        elif id_type == "organ":
            token_aligner = organ_align_mapping
            all_tokens = organ_voc
        else:
            raise Exception(f"{id_type} is not a valid id type")
        self.token2id = {token: i for i, token in enumerate(all_tokens)}
        self.id2token = {i: token for token, i in self.token2id.items()}
        self.token_aligner = token_aligner

    def tokenize(self, token):
        if token not in self.token2id:
            return self.token2id["<unk>"]
        return self.token2id[token]

    def align(self, input):
        if isinstance(input, str):
            return self.token_aligner[input] if input in self.token_aligner else "<pad>"
        elif isinstance(input, list):
            return [
                self.token_aligner[token] if token in self.token_aligner else "<pad>"
                for token in ["Visium", "Xenium", "VisiumHD"]
            ]

    def encode(self, input, align_first=False):
        if align_first:
            input = self.align(input)

        if isinstance(input, str):
            return self.token2id[input]
        elif isinstance(input, list):
            return [self.token2id[token] for token in input]

    def decode(self, input):
        if isinstance(input, int):
            return self.id2token[input]
        elif isinstance(input, list):
            return [self.id2token[i] for i in input]

    @property
    def n_tokens(self):
        return len(self.token2id)

    @property
    def mask_token(self) -> str:
        return "<mask>"

    @property
    def mask_token_id(self) -> int:
        return self.token2id[self.mask_token]

    @property
    def pad_token(self) -> str:
        return "<pad>"

    @property
    def pad_token_id(self) -> int:
        return self.token2id[self.pad_token]

    @property
    def unk_token(self) -> str:
        return "<unk>"

    @property
    def unk_token_id(self) -> int:
        return self.token2id[self.unk_token]


class AnnotationTokenizer(TokenizerBase):
    def __init__(
        self,
        id_type="disease",
    ):
        if id_type == "disease":
            token_aligner = cancer_annotation_align_mapping
            all_tokens = cancer_annotation_voc
        elif id_type == "domain":
            token_aligner = domain_annotation_align_mapping
            all_tokens = domain_annotation_voc
        else:
            raise Exception(f"{id_type} is not a valid id type")
        self.token2id = {token: i for i, token in enumerate(all_tokens)}
        self.token_aligner = token_aligner

    def tokenize(self, token):
        if token not in self.token2id:
            return self.token2id["<unk>"]
        return self.token2id[token]

    def align(self, input):
        if isinstance(input, str):
            return self.token_aligner[input] if input in self.token_aligner else "<unk>"
        elif isinstance(input, list):
            return [
                self.token_aligner[token] if token in self.token_aligner else "<unk>"
                for token in input
            ]

    def encode(self, input, align_first=False):
        if align_first:
            input = self.align(input)

        if isinstance(input, str):
            return self.token2id[input]
        elif isinstance(input, list):
            return [self.token2id[token] for token in input]

    @property
    def n_tokens(self):
        return len(self.token2id)

    @property
    def mask_token(self) -> str:
        return self.token2id["<mask>"]

    @property
    def pad_token(self) -> str:
        return self.token2id["<pad>"]

    @property
    def unk_token(self) -> str:
        return self.token2id["<unk>"]


class TokenizerTools:
    ge_tokenizer: GeneExpTokenizer | None = None
    image_tokenizer: ImageTokenizer | None = None
    tech_tokenizer: IDTokenizer | None = None
    specie_tokenizer: IDTokenizer | None = None
    organ_tokenizer: IDTokenizer | None = None
    cancer_anno_tokenizer: AnnotationTokenizer | None = None
    domain_anno_tokenizer: AnnotationTokenizer | None = None

    def __init__(
        self,
        ge_tokenizer,
        image_tokenizer,
        tech_tokenizer,
        specie_tokenizer,
        organ_tokenizer,
        cancer_anno_tokenizer,
        domain_anno_tokenizer,
        **kwargs,
    ):
        self.ge_tokenizer = ge_tokenizer
        self.image_tokenizer = image_tokenizer
        self.tech_tokenizer = tech_tokenizer
        self.specie_tokenizer = specie_tokenizer
        self.organ_tokenizer = organ_tokenizer
        self.cancer_anno_tokenizer = cancer_anno_tokenizer
        self.domain_anno_tokenizer = domain_anno_tokenizer

        for k, v in kwargs.items():
            setattr(self, k, v)


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


class InputEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.image_embed = nn.Linear(config["d_input"], config["d_model"])
        self.gene_embed = nn.Linear(config["n_genes"], config["d_model"], bias=False)
        self.tech_embed = nn.Embedding(config["n_tech"], config["d_model"])
        self.organ_embed = nn.Embedding(config["n_organs"], config["d_model"])

    def forward(
        self,
        img_tokens,
        ge_tokens,
        tech_tokens,
        organ_tokens,
    ):
        img_embed = self.image_embed(img_tokens)
        ge_embed = self.gene_embed(ge_tokens)
        tech_embed = self.tech_embed(tech_tokens)
        organ_embed = self.organ_embed(organ_tokens)

        return img_embed + ge_embed + tech_embed + organ_embed


class SpatialTransformer(nn.Module):
    def __init__(self, config):
        super(SpatialTransformer, self).__init__()

        self.blks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=config["d_model"],
                    n_heads=config["n_heads"],
                    activation=config["act"],
                    attn_drop=config["attn_dropout"],
                    proj_drop=config["dropout"],
                    mlp_ratio=config["mlp_ratio"],
                )
                for i in range(config["n_layers"])
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


class STFM(nn.Module):
    def __init__(self, config) -> None:
        super(STFM, self).__init__()

        self.model = SpatialTransformer(config)
        self.input_encoder = InputEncoder(config)

        self.gene_exp_head = nn.Sequential(
            nn.LayerNorm(config["d_model"]),
            nn.Linear(config["d_model"], config["n_genes"]),
        )

    def inference(
        self,
        img_tokens: torch.Tensor,
        coords: torch.Tensor,
        ge_tokens: torch.Tensor,
        batch_idx: torch.Tensor,
        tech_tokens: torch.Tensor | None = None,
        organ_tokens: torch.Tensor | None = None,
    ):
        x = self.input_encoder(
            img_tokens=img_tokens,
            ge_tokens=ge_tokens,
            tech_tokens=tech_tokens,
            organ_tokens=organ_tokens,
        )
        return self.model(x, coords, batch_idx)

    # def prediction_head(
    def forward(
        self,
        img_tokens: torch.Tensor,
        coords: torch.Tensor,
        ge_tokens: torch.Tensor,
        batch_idx: torch.Tensor,
        tech_tokens: torch.Tensor | None = None,
        organ_tokens: torch.Tensor | None = None,
        return_all=False,
    ):
        x = self.inference(
            img_tokens=img_tokens,
            coords=coords,
            batch_idx=batch_idx,
            ge_tokens=ge_tokens,
            tech_tokens=tech_tokens,
            organ_tokens=organ_tokens,
        )

        if return_all:
            return self.gene_exp_head(x), x
        return self.gene_exp_head(x)


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
    def __init__(self, model_path=None, gene_voc_path=None, token=None):
        from huggingface_hub import hf_hub_download

        if model_path is None:
            model_path = hf_hub_download(
                "tlhuang/STPath",
                "stfm.pth",
            )

        if gene_voc_path is None:
            gene_voc_path = (
                "/Users/simon/rep/lbi/reindero/STPath/utils_data/symbol2ensembl.json"
            )

        config = {
            "d_input": 1536,
            "d_model": 512,
            "n_layers": 4,
            "n_heads": 4,
            "dropout": 0.1,
            "attn_dropout": 0.1,
            "act": "gelu",
            "mlp_ratio": 2.0,
        }

        self.tokenizer = TokenizerTools(
            ge_tokenizer=GeneExpTokenizer(gene_voc_path),
            image_tokenizer=ImageTokenizer(feature_dim=1536),
            tech_tokenizer=IDTokenizer(id_type="tech"),
            specie_tokenizer=IDTokenizer(id_type="specie"),
            organ_tokenizer=IDTokenizer(id_type="organ"),
            cancer_anno_tokenizer=AnnotationTokenizer(id_type="disease"),
            domain_anno_tokenizer=AnnotationTokenizer(id_type="domain"),
        )

        config["n_genes"] = self.tokenizer.ge_tokenizer.n_tokens
        config["n_tech"] = self.tokenizer.tech_tokenizer.n_tokens
        config["n_species"] = self.tokenizer.specie_tokenizer.n_tokens
        config["n_organs"] = self.tokenizer.organ_tokenizer.n_tokens
        config["n_cancer_annos"] = self.tokenizer.cancer_anno_tokenizer.n_tokens
        config["n_domain_annos"] = self.tokenizer.domain_anno_tokenizer.n_tokens

        print(config)

        self.stfm = STFM(config=config)
        self.stfm.load_state_dict(
            torch.load(model_path, map_location="cpu"), strict=True
        )

        print(f"Model loaded from {model_path}")
        self.stfm.eval()

    def predict(self, image):
        return None
