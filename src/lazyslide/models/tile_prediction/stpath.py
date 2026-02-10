"""
This module defines the STPath model, tokenizers, and related components for
spatial transcriptomics data processing and tile prediction.

It includes:
- Tokenizer interfaces and implementations for gene expression, image features,
  and metadata (technology, species, organ, cancer annotation, domain annotation).
- Transformer-based neural network blocks for spatial feature learning.
- The main STFM (Spatial Transcriptomics Foundation Model) class.
- The STPath class, which integrates the tokenizers and STFM for tile prediction.
"""

import json
from typing import Optional, Protocol, runtime_checkable

import numpy as np
import scanpy as sc
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from torch import Tensor, nn

from .._model_registry import register
from ..base import ModelBase, ModelTask

# lookup tables
# technology
tech_align_mapping = {
    "ST": "Spatial Transcriptomics",
    "Visium": "Visium",
    "Xenium": "Xenium",
    "VisiumHD": "Visium HD",
    "Visium HD": "Visium HD",
    "Spatial Transcriptomics": "Spatial Transcriptomics",
}
"""
Mapping for aligning various technology identifiers to a standardized vocabulary.
Keys are raw technology names, values are standardized technology names.
"""
tech_voc = ["<pad>", "Spatial Transcriptomics", "Visium", "Xenium", "Visium HD"]
"""
Vocabulary for spatial transcriptomics technologies, including special tokens.
"""


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
"""
Mapping for aligning various species identifiers to a standardized vocabulary.
Keys are raw species names, values are standardized species names.
"""
specie_voc = ["<pad>", "<mask>", "<unk>", "Mus musculus", "Homo sapiens", "others"]
"""
Vocabulary for species, including special tokens.
"""


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
"""
Mapping for aligning various organ identifiers to a standardized vocabulary.
Keys are raw organ names, values are standardized organ names.
"""
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
"""
Vocabulary for organs, including special tokens.
"""

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
"""
Mapping for aligning various cancer annotation terms to a standardized vocabulary
of 'tumor' or 'healthy'.
"""
cancer_annotation_voc = ["<pad>", "<mask>", "<unk>", "healthy", "tumor"]
"""
Vocabulary for cancer annotations, including special tokens.
"""

domain_annotation_align_mapping = {
    "l1": "l1",
    "l2": "l2",
    "l3": "l3",
    "l4": "l4",
    "l5": "l5",
    "l6": "l6",
    "vm": "vm",
}
"""
Mapping for aligning various brain layer (domain) annotation terms to a standardized vocabulary.
"""

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
"""
Vocabulary for brain layer (domain) annotations, including special tokens.
"""


@runtime_checkable
class TokenizerBase(Protocol):
    """
    A protocol defining the interface for all tokenizer classes.
    Ensures that all tokenizers have common methods and properties.
    """

    def encode(self, *args, **kwargs):
        """Encodes input data into token IDs or features."""
        ...

    def decode(self, *args, **kwargs):
        """Decodes token IDs or features back into original representation."""
        ...

    @property
    def mask_token(self):
        """Returns the mask token."""
        ...

    @property
    def mask_token_id(self) -> int:
        """Returns the ID of the mask token."""
        ...

    @property
    def pad_token(self):
        """Returns the pad token."""
        ...

    @property
    def pad_token_id(self) -> int:
        """Returns the ID of the pad token."""
        ...


class GeneExpTokenizer(TokenizerBase):
    """
    A tokenizer specifically designed for gene expression data.
    It maps gene symbols to unique integer IDs and handles one-hot encoding.
    """

    def __init__(
        self,
        symbol2gene_path: str,
    ):
        """
        Initializes the GeneExpTokenizer.

        Args:
            symbol2gene_path (str): Path to a JSON file mapping gene symbols to gene IDs.
        """
        self.symbol2gene = json.load(open(symbol2gene_path, "r"))
        gene2id = list(set(self.symbol2gene.values()))
        gene2id.sort()  # make sure the order is fixed
        self.gene2id = {gene_id: i for i, gene_id in enumerate(gene2id)}

        # add 2 for pad and mask token
        # Token IDs are shifted by 2 to reserve 0 for <pad> and 1 for <mask>.
        self.gene2id = {gene: token_id + 2 for gene, token_id in self.gene2id.items()}

    def get_available_symbols(self) -> list[str]:
        """
        Returns a list of gene symbols that are available in the tokenizer.
        """
        return [
            g
            for g in list(self.symbol2gene.keys())
            if self.symbol2gene[g] in self.gene2id
        ]

    def get_available_genes(self) -> list[str]:
        """
        Returns a list of gene IDs (from symbol2gene_path) that the tokenizer can encode,
        ranked by their internal token ID.
        """
        # rank the genes by their id
        return sorted(self.gene2id.keys(), key=lambda x: self.gene2id[x])

    def convert_gene_exp_to_one_hot_tensor(
        self, n_genes: int, gene_exp: Tensor, gene_ids: Tensor
    ) -> Tensor:
        """
        Converts gene expression values to a one-hot tensor representation.

        Args:
            n_genes (int): Total number of genes (including special tokens).
            gene_exp (Tensor): Gene expression values [N_cells, N_genes].
            gene_ids (Tensor): Tensor of gene IDs corresponding to the gene_exp columns [N_genes].

        Returns:
            Tensor: One-hot encoded gene expression tensor [N_cells, n_genes].
        """
        # gene_exp: [N_cells, N_genes], gene_ids: [N_genes]
        # Expand gene_ids to match the batch size of gene_exp, then clamp to valid range.
        gene_ids = gene_ids.clamp(min=0, max=n_genes)[None, ...].expand(
            gene_exp.shape[0], -1
        )
        # Create a zero tensor for one-hot representation.
        gene_exp_onehot = torch.zeros(gene_exp.size(0), n_genes, device=gene_exp.device)
        # Scatter gene_exp values into the one-hot tensor at specified gene_ids.
        gene_exp_onehot.scatter_(dim=1, index=gene_ids, src=gene_exp)
        return gene_exp_onehot

    def symbol2id(self, symbol_list: list[str], return_valid_positions: bool = False):
        """
        Converts a list of gene symbols to their corresponding token IDs.

        Args:
            symbol_list (list[str]): A list of gene symbols.
            return_valid_positions (bool): If True, also returns the original positions
                                            of the valid symbols in the input list.

        Returns:
            Union[list[int], Tuple[list[int], list[int]]]: A list of token IDs,
            and optionally, the valid positions.
        """
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

    def shift_token_id(self, n_shift: int):
        """
        Shifts all gene token IDs by a specified amount.
        This is useful when combining tokenizers or reserving IDs for special tokens.

        Args:
            n_shift (int): The amount to shift token IDs by.
        """
        self.gene2id = {
            gene: token_id + n_shift for gene, token_id in self.gene2id.items()
        }

    def subset_gene_ids(self, symbol_list: list[str]):
        """
        Subsets the tokenizer's internal gene ID mapping to only include genes
        present in the given symbol list.

        Args:
            symbol_list (list[str]): A list of gene symbols to keep.
        """
        gene_ids = [self.symbol2gene[s] for s in symbol_list if s in self.symbol2gene]
        # only keep the valid gene ids in self.gene2ids
        self.gene2id = {gene_id: i for i, gene_id in enumerate(gene_ids)}

    def get_hvg(self, adata: sc.AnnData, n_top_genes: int = 100):
        """
        Identifies highly variable genes (HVGs) from an AnnData object
        and returns their names and token IDs.

        Args:
            adata (sc.AnnData): An AnnData object containing gene expression data.
            n_top_genes (int): The number of top highly variable genes to select.

        Returns:
            Tuple[list[str], Tensor]: A tuple containing a list of HVG names
                                      and a Tensor of their corresponding token IDs.
        """
        if "raw_hvg_names" in adata.uns:
            hvg_names = adata.uns["raw_hvg_names"][:n_top_genes]
        else:
            all_gene_names = set(self.get_available_symbols())
            available_genes = [
                g for g in adata.var_names.tolist() if g in all_gene_names
            ]
            adata = adata.copy()[:, available_genes]

            # Preprocessing for highly variable gene selection
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

    def encode(self, adata: sc.AnnData, return_sparse: bool = False):
        """
        Encodes gene expression data from an AnnData object into a tensor format.

        Args:
            adata (sc.AnnData): An AnnData object containing gene expression data.
            return_sparse (bool): If True, returns a sparse tensor representation.
                                  Otherwise, returns a dense one-hot representation.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the encoded gene expression tensor
                                   and a tensor of observed gene IDs.
        """
        gene_list = [
            g
            for g in adata.var_names.tolist()
            if g in self.symbol2gene
            and self.symbol2gene[g] is not None
            and self.symbol2gene[g] in self.gene2id
        ]
        # Filter AnnData object to include only genes known to the tokenizer
        adata = adata[:, gene_list]

        if not isinstance(adata.X, np.ndarray):
            # Handle sparse matrix format (e.g., scipy.sparse.csr_matrix)
            counts_per_row = np.diff(adata.X.indptr)
            row = np.repeat(
                np.arange(adata.X.shape[0]), counts_per_row
            )  # create row indices for the sparse matrix
            col = adata.X.indices
            act_gene_exp = adata.X.data  # nonzero values
        else:
            # Handle dense matrix format (e.g., numpy.ndarray)
            row, col = np.nonzero(adata.X)
            act_gene_exp = adata.X[row, col]

        # Map observed gene symbols to their token IDs
        # TODO: there might be some genes sharing the same id so some expression values will be missing
        obs_gene_ids = np.array(
            [self.gene2id[self.symbol2gene[g]] for g in adata.var_names.tolist()]
        )
        col = obs_gene_ids[col]  # Adjust column indices to reflect token IDs
        obs_gene_ids = torch.from_numpy(obs_gene_ids).long()
        act_gene_exp = torch.from_numpy(act_gene_exp).float()

        indices = torch.stack(
            [torch.from_numpy(row), torch.from_numpy(col)], dim=0
        ).long()

        if return_sparse:
            try:
                from torch_geometric.utils import coalesce
            except ImportError:
                print(
                    "torch_geometric is not installed. Falling back to dense representation."
                )
                return_sparse = False  # Fallback if torch_geometric is not available
            if return_sparse:
                # remove duplicate indices by coalescing
                # 'max' reduction picks the maximum value if there are multiple values for the same edge
                indices, act_gene_exp = coalesce(
                    edge_index=indices,
                    edge_attr=act_gene_exp,
                    reduce="max",
                )
                gene_exp = torch.sparse_coo_tensor(
                    indices, act_gene_exp, size=(adata.shape[0], self.n_tokens)
                )
        if not return_sparse:  # If return_sparse was False initially or fell back
            gene_exp = adata.to_df().values
            gene_exp = torch.from_numpy(gene_exp).float()
            gene_exp = self.convert_gene_exp_to_one_hot_tensor(
                self.n_tokens, gene_exp, obs_gene_ids
            )

        # gene_exp: [n_cells, n_genes], obs_gene_ids: [-1]
        return gene_exp, obs_gene_ids

    @property
    def n_tokens(self) -> int:
        """Returns the total number of unique tokens (including special tokens)."""
        return max(self.gene2id.values()) + 1

    @property
    def mask_token(self):
        """
        Returns the one-hot representation of the mask token.
        This is a vector of zeros with a 1 at the mask_token_id position.
        """
        return F.one_hot(torch.tensor(1), num_classes=self.n_tokens).float()

    @property
    def mask_token_id(self) -> int:
        """Returns the integer ID for the mask token."""
        return 1

    @property
    def pad_token(self):
        """
        Returns the one-hot representation of the padding token.
        This is a vector of zeros with a 1 at the pad_token_id position.
        """
        return F.one_hot(torch.tensor(0), num_classes=self.n_tokens).float()

    @property
    def pad_token_id(self) -> int:
        """Returns the integer ID for the padding token."""
        return 0


# To accomodate the image features,
# the mask token and pad token are defined as one-hot vectors
# with the same dimension as the image features
class ImageTokenizer(TokenizerBase):
    """
    A tokenizer for image features. The mask and pad tokens are represented
    as one-hot vectors with the same dimension as the image features.
    """

    def __init__(
        self,
        feature_dim: int,
    ):
        """
        Initializes the ImageTokenizer.

        Args:
            feature_dim (int): The dimension of the image features (e.g., 1024).
        """
        super().__init__()

        self.feature_dim = feature_dim  # dimension of image features, e.g., 1024

    @property
    def n_tokens(self) -> int:
        """
        Returns the number of special tokens.
        For ImageTokenizer, this typically refers to mask and pad.
        """
        return 2

    @property
    def mask_token(self) -> Tensor:
        """
        Returns the one-hot representation of the mask token.
        It's a vector of zeros with a 1 at the mask_token_id position within feature_dim.
        """
        # The mask token is a one-hot vector in the feature dimension
        return F.one_hot(torch.tensor(1), num_classes=self.feature_dim).float()

    @property
    def mask_token_id(self) -> int:
        """Returns the integer ID for the mask token."""
        return 1

    @property
    def pad_token(self) -> Tensor:
        """
        Returns the one-hot representation of the padding token.
        It's a vector of zeros with a 1 at the pad_token_id position within feature_dim.
        """
        # The pad token is a one-hot vector in the feature dimension
        return F.one_hot(torch.tensor(0), num_classes=self.feature_dim).float()

    @property
    def pad_token_id(self) -> int:
        """Returns the integer ID for the padding token."""
        return 0


class IDTokenizer(TokenizerBase):
    """
    A generic tokenizer for categorical identifiers (e.g., technology, species, organ).
    It maps string tokens to unique integer IDs based on a predefined vocabulary.
    """

    def __init__(
        self,
        id_type: str,
    ):
        """
        Initializes the IDTokenizer.

        Args:
            id_type (str): The type of identifier to tokenize.
                           Supported types: "tech", "specie", "organ".
        """
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

    def tokenize(self, token: str) -> int:
        """
        Converts a single token string to its corresponding ID.
        If the token is not in the vocabulary, it returns the ID for '<unk>'.

        Args:
            token (str): The token string to convert.

        Returns:
            int: The integer ID of the token.
        """
        if token not in self.token2id:
            return self.token2id["<unk>"]
        return self.token2id[token]

    def align(self, input_data):
        """
        Aligns input tokens to the standardized vocabulary using the predefined mapping.

        Args:
            input_data (Union[str, list[str]]): A single token string or a list of token strings.

        Returns:
            Union[str, list[str]]: The aligned token(s).
        """
        if isinstance(input_data, str):
            return (
                self.token_aligner[input_data]
                if input_data in self.token_aligner
                else "<pad>"
            )
        elif isinstance(input_data, list):
            # This part of the code seems to have a hardcoded list ["Visium", "Xenium", "VisiumHD"]
            # It might be intended to align a predefined set of technologies rather than the input list.
            # Depending on the intended behavior, this might need adjustment.
            return [
                self.token_aligner[token] if token in self.token_aligner else "<pad>"
                for token in ["Visium", "Xenium", "VisiumHD"]
            ]

    def encode(self, input_data, align_first: bool = False):
        """
        Encodes input token(s) to their corresponding IDs.

        Args:
            input_data (Union[str, list[str]]): A single token string or a list of token strings.
            align_first (bool): If True, the input tokens are first aligned using `align()`
                                before encoding.

        Returns:
            Union[int, list[int]]: The integer ID(s) of the token(s).
        """
        if align_first:
            input_data = self.align(input_data)

        if isinstance(input_data, str):
            return self.token2id[input_data]
        elif isinstance(input_data, list):
            return [self.token2id[token] for token in input_data]

    def decode(self, input_ids):
        """
        Decodes token IDs back to their string representations.

        Args:
            input_ids (Union[int, list[int]]): A single token ID or a list of token IDs.

        Returns:
            Union[str, list[str]]: The string representation(s) of the token ID(s).
        """
        if isinstance(input_ids, int):
            return self.id2token[input_ids]
        elif isinstance(input_ids, list):
            return [self.id2token[i] for i in input_ids]

    @property
    def n_tokens(self) -> int:
        """Returns the total number of unique tokens in the vocabulary."""
        return len(self.token2id)

    @property
    def mask_token(self) -> str:
        """Returns the string representation of the mask token."""
        return "<mask>"

    @property
    def mask_token_id(self) -> int:
        """Returns the integer ID for the mask token."""
        return self.token2id[self.mask_token]

    @property
    def pad_token(self) -> str:
        """Returns the string representation of the padding token."""
        return "<pad>"

    @property
    def pad_token_id(self) -> int:
        """Returns the integer ID for the padding token."""
        return self.token2id[self.pad_token]

    @property
    def unk_token(self) -> str:
        """Returns the string representation of the unknown token."""
        return "<unk>"

    @property
    def unk_token_id(self) -> int:
        """Returns the integer ID for the unknown token."""
        return self.token2id[self.unk_token]


class AnnotationTokenizer(TokenizerBase):
    """
    A tokenizer for annotation data (e.g., cancer, domain).
    It maps annotation terms to unique integer IDs.
    """

    def __init__(
        self,
        id_type: str,
    ):
        """
        Initializes the AnnotationTokenizer.

        Args:
            id_type (str): The type of annotation to tokenize.
                           Supported types: "disease", "domain".
        """
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

    def tokenize(self, token: str) -> int:
        """
        Converts a single annotation token string to its corresponding ID.
        If the token is not in the vocabulary, it returns the ID for '<unk>'.

        Args:
            token (str): The annotation token string to convert.

        Returns:
            int: The integer ID of the token.
        """
        if token not in self.token2id:
            return self.token2id["<unk>"]
        return self.token2id[token]

    def align(self, input_data):
        """
        Aligns input annotation terms to the standardized vocabulary.

        Args:
            input_data (Union[str, list[str]]): A single annotation term or a list of terms.

        Returns:
            Union[str, list[str]]: The aligned annotation term(s).
        """
        if isinstance(input_data, str):
            return (
                self.token_aligner[input_data]
                if input_data in self.token_aligner
                else "<unk>"
            )
        elif isinstance(input_data, list):
            return [
                self.token_aligner[token] if token in self.token_aligner else "<unk>"
                for token in input_data
            ]

    def encode(self, input_data, align_first: bool = False):
        """
        Encodes input annotation term(s) to their corresponding IDs.

        Args:
            input_data (Union[str, list[str]]): A single annotation term or a list of terms.
            align_first (bool): If True, the input tokens are first aligned using `align()`
                                before encoding.

        Returns:
            Union[int, list[int]]: The integer ID(s) of the annotation term(s).
        """
        if align_first:
            input_data = self.align(input_data)

        if isinstance(input_data, str):
            return self.token2id[input_data]
        elif isinstance(input_data, list):
            return [self.token2id[token] for token in input_data]

    @property
    def n_tokens(self) -> int:
        """Returns the total number of unique tokens in the vocabulary."""
        return len(self.token2id)

    @property
    def mask_token(self) -> int:
        """Returns the integer ID for the mask token."""
        return self.token2id["<mask>"]

    @property
    def pad_token(self) -> int:
        """Returns the integer ID for the padding token."""
        return self.token2id["<pad>"]

    @property
    def unk_token(self) -> int:
        """Returns the integer ID for the unknown token."""
        return self.token2id["<unk>"]


class TokenizerTools:
    """
    A container class to hold and manage various tokenizer instances.
    This provides a single point of access for all tokenization needs within the model.
    """

    ge_tokenizer: GeneExpTokenizer
    image_tokenizer: ImageTokenizer
    tech_tokenizer: IDTokenizer
    specie_tokenizer: IDTokenizer
    organ_tokenizer: IDTokenizer
    cancer_anno_tokenizer: AnnotationTokenizer
    domain_anno_tokenizer: AnnotationTokenizer

    def __init__(
        self,
        ge_tokenizer: GeneExpTokenizer,
        image_tokenizer: ImageTokenizer,
        tech_tokenizer: IDTokenizer,
        specie_tokenizer: IDTokenizer,
        organ_tokenizer: IDTokenizer,
        cancer_anno_tokenizer: AnnotationTokenizer,
        domain_anno_tokenizer: AnnotationTokenizer,
    ):
        """
        Initializes TokenizerTools with instances of different tokenizers.

        Args:
            ge_tokenizer (GeneExpTokenizer): Tokenizer for gene expression data.
            image_tokenizer (ImageTokenizer): Tokenizer for image features.
            tech_tokenizer (IDTokenizer): Tokenizer for technology identifiers.
            specie_tokenizer (IDTokenizer): Tokenizer for species identifiers.
            organ_tokenizer (IDTokenizer): Tokenizer for organ identifiers.
            cancer_anno_tokenizer (AnnotationTokenizer): Tokenizer for cancer annotations.
            domain_anno_tokenizer (AnnotationTokenizer): Tokenizer for domain annotations.
        """
        self.ge_tokenizer = ge_tokenizer
        self.image_tokenizer = image_tokenizer
        self.tech_tokenizer = tech_tokenizer
        self.specie_tokenizer = specie_tokenizer
        self.organ_tokenizer = organ_tokenizer
        self.cancer_anno_tokenizer = cancer_anno_tokenizer
        self.domain_anno_tokenizer = domain_anno_tokenizer


class TransformerBlock(nn.Module):
    """
    A single block of a transformer model, incorporating attention and an MLP.
    It applies self-attention to token embeddings considering spatial coordinates,
    followed by a feed-forward neural network.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 1,
        activation: str = "gelu",
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mlp_ratio: float = 4.0,
    ):
        """
        Initializes a TransformerBlock.

        Args:
            d_model (int): Dimension of the model's latent space.
            n_heads (int): Number of attention heads.
            activation (str): Activation function to use in the MLP ("gelu", "silu", "relu", "swiglu").
            attn_drop (float): Dropout rate for attention weights.
            proj_drop (float): Dropout rate for the projection layers.
            mlp_ratio (float): Ratio of MLP hidden dimension to model dimension.
        """
        super(TransformerBlock, self).__init__()

        self.attn = Attention(
            d_model=d_model, n_heads=n_heads, proj_drop=proj_drop, attn_drop=attn_drop
        )

        self.mlp = MLPWrapper(
            in_features=d_model,
            hidden_features=int(d_model * mlp_ratio),
            out_features=d_model,
            activation=activation,
            norm_layer=nn.LayerNorm,
            drop=proj_drop,  # Corrected: 'drop' parameter is now passed
        )

    def forward(
        self, token_embs: Tensor, coords: Tensor, padding_mask: Tensor
    ) -> Tensor:
        """
        Forward pass for the TransformerBlock.

        Args:
            token_embs (Tensor): Input token embeddings [Batch, N_tokens, D_model].
            coords (Tensor): Spatial coordinates of the tokens [Batch, N_tokens, 2].
            padding_mask (Tensor): Mask to ignore padded tokens [Batch, N_tokens, N_tokens].

        Returns:
            Tensor: Output token embeddings after applying attention and MLP [Batch, N_tokens, D_model].
        """
        context_token_embs = self.attn(token_embs, coords, padding_mask)
        token_embs = token_embs + context_token_embs

        token_embs = token_embs + self.mlp(token_embs)
        return token_embs


class FrameAveraging(nn.Module):
    """
    Implements Frame Averaging for rotation invariance in spatial data.
    It creates multiple rotated "frames" of the input coordinates, applies operations,
    and then averages the results to achieve rotation invariance.
    """

    def __init__(self, dim: int = 3, backward: bool = False):
        """
        Initializes FrameAveraging.

        Args:
            dim (int): The dimension of the input coordinates (e.g., 2 for 2D, 3 for 3D).
            backward (bool): If True, allows gradients to flow through frame creation.
        """
        super(FrameAveraging, self).__init__()

        self.dim = dim
        self.n_frames = 2**dim  # Number of frames (rotations) to consider
        self.ops = self.create_ops(dim)  # [2^dim, dim] - Rotation operations
        self.backward = backward

    def create_ops(self, dim: int) -> Tensor:
        """
        Generates the rotation operations (flips along each dimension).

        Args:
            dim (int): The dimension of the space.

        Returns:
            Tensor: A tensor of shape (2^dim, dim) representing the flip operations.
        """
        colon = slice(None)
        accum = []
        directions = torch.tensor([-1, 1])

        for ind in range(dim):
            dim_slice = [None] * dim
            dim_slice[ind] = colon
            accum.append(directions[dim_slice])

        accum = torch.broadcast_tensors(*accum)
        operations = torch.stack(accum, dim=-1)
        operations = operations.flatten(
            0, -2
        )  # Equivalent to rearrange(operations, "... d -> (...) d")
        return operations

    def create_frame(self, X: Tensor, mask: Optional[Tensor]):
        """
        Creates rotated frames of the input coordinates and calculates transformation matrices.

        Args:
            X (Tensor): Input coordinates [Batch, N_points, Dim].
            mask (Optional[Tensor]): Mask indicating valid points [Batch, N_points].

        Returns:
            Tuple[Tensor, Tensor, Tensor]:
                - h: Transformed coordinates in multiple frames [Batch * N_frames, N_points, Dim].
                - F_ops: Detached transformation matrices [Batch, N_frames, Dim, Dim].
                - center: Centroid of the points for recentering [Batch, Dim].
        """
        assert X.shape[-1] == self.dim, (
            f"expected points of dimension {self.dim}, but received {X.shape[-1]}"
        )

        if mask is None:
            b, n, _ = X.shape
            mask = torch.ones(b, n, device=X.device).to(torch.bool)
        mask = mask.unsqueeze(-1)  # [B, N, 1]
        center = (X * mask).sum(dim=1) / mask.sum(dim=1)  # Calculate centroid
        X = X - center.unsqueeze(1) * mask  # [B,N,dim] - Recenter points
        X_ = X.masked_fill(~mask, 0.0)  # Mask out invalid points

        C = torch.bmm(X_.transpose(1, 2), X_)  # [B,dim,dim] (Covariance matrix)
        if not self.backward:
            C = C.detach()  # Detach if gradients are not needed for frame creation

        # Eigh returns eigenvalues and eigenvectors
        _, eigenvectors = torch.linalg.eigh(C, UPLO="U")  # [B,dim,dim] - Principal axes
        # Apply rotation operations (flips) to the eigenvectors
        F_ops = self.ops.unsqueeze(1).unsqueeze(0).to(
            X.device
        ) * eigenvectors.unsqueeze(1)
        # [1,2^dim,1,dim] x [B,1,dim,dim] -> [B,2^dim,dim,dim]
        # F_ops contains the transformation matrices for each frame

        # Apply transformations to the points
        h = torch.einsum(
            "boij,bpj->bopi", F_ops.transpose(2, 3), X
        )  # transpose is inverse [B,2^dim,N,dim]

        h = h.view(X.size(0) * self.n_frames, X.size(1), self.dim)
        return h, F_ops.detach(), center

    def invert_frame(
        self, X: Tensor, mask: Optional[Tensor], F_ops: Tensor, center: Tensor
    ) -> Tensor:
        """
        Inverts the frame averaging transformation to get back to the original coordinate system.

        Args:
            X (Tensor): Transformed coordinates in multiple frames.
            mask (Optional[Tensor]): Mask indicating valid points.
            F_ops (Tensor): Transformation matrices used during frame creation.
            center (Tensor): Centroid used during recentering.

        Returns:
            Tensor: Coordinates in the original frame [Batch, N_points, Dim].
        """
        # Apply inverse transformation
        X = torch.einsum("boij,bopj->bopi", F_ops, X)
        X = X.mean(dim=1)  # frame averaging
        X = X + center.unsqueeze(1)  # Add back the centroid
        if mask is None:
            return X
        return X * mask.unsqueeze(-1)


@torch.jit.script
def transform_qkv(
    q: Tensor, k: Tensor, v: Tensor, n_heads: int
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Transforms Q, K, V tensors from [Batch, Sequence Length, (Heads * Head Dimension)]
    to [Batch, Heads, Sequence Length, Head Dimension].
    This reshaping is essential for multi-head attention computations.
    Equivalent to einops.rearrange(x, "b n (h d) -> b h n d", h=n_heads).

    Args:
        q (Tensor): Query tensor.
        k (Tensor): Key tensor.
        v (Tensor): Value tensor.
        n_heads (int): Number of attention heads.

    Returns:
        tuple[Tensor, Tensor, Tensor]: Transformed Q, K, V tensors.
    """
    # Get dimensions
    # B: Batch, N: Sequence Length, Total_Dim: h * d
    b, n, total_dim = q.shape
    d = total_dim // n_heads  # Head dimension

    # Reshape to [B, N, H, D]
    q = q.view(b, n, n_heads, d)
    k = k.view(b, n, n_heads, d)
    v = v.view(b, n, n_heads, d)

    # Permute to [B, H, N, D] to align with multi-head attention requirements
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    return q, k, v


class Attention(FrameAveraging):
    """
    Multi-head attention mechanism with spatial bias derived from Frame Averaging.
    This attention mechanism incorporates relative spatial information into the
    attention scores.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 1,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        max_n_tokens: int = 5000,  # Typo: was 5e3
    ):
        """
        Initializes the Attention module.

        Args:
            d_model (int): Dimension of the model's latent space.
            n_heads (int): Number of attention heads.
            proj_drop (float): Dropout rate for the projection layers.
            attn_drop (float): Dropout rate for attention weights.
            max_n_tokens (int): Maximum number of tokens (used for potential future optimizations, currently not directly used).
        """
        super(Attention, self).__init__(dim=2)  # FrameAveraging for 2D coordinates

        self.max_n_tokens = max_n_tokens
        self.d_head = d_model // n_heads  # Dimension of each attention head
        self.n_heads = n_heads
        self.scale = self.d_head**-0.5  # Scaling factor for attention scores

        self.layernorm_qkv = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 3),  # Projects input to Q, K, V
        )
        self.W_output = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(proj_drop),
        )
        self.attn_dropout = nn.Dropout(attn_drop)
        # Linear layer to compute spatial bias from radial coordinates and norms
        self.edge_bias = nn.Sequential(
            nn.Linear(
                self.dim + 1, self.n_heads, bias=False
            ),  # Input: [coord_dim + norm_dim]
        )

    def forward(self, x: Tensor, coords: Tensor, pad_mask: Optional[Tensor]) -> Tensor:
        """
        Forward pass for the Attention module.

        Args:
            x (Tensor): Input tensor of token embeddings [Batch, N_tokens, D_model].
            coords (Tensor): Spatial coordinates of the tokens [Batch, N_tokens, 2].
            pad_mask (Optional[Tensor]): Padding mask [Batch, N_tokens, N_tokens].
                                        True indicates padded elements to be ignored.

        Returns:
            Tensor: Output tensor after applying attention [Batch, N_tokens, D_model].
        """
        B, N, C = x.shape
        q, k, v = self.layernorm_qkv(x).chunk(3, dim=-1)  # Project to Q, K, V
        # Reshape Q, K, V for multi-head attention
        q, k, v = transform_qkv(q, k, v, self.n_heads)

        q = q * self.scale  # Scale Q
        attn = q @ k.transpose(-2, -1)  # Compute raw attention scores [B, H, N, N]

        """Build pairwise representation with Frame Averaging"""
        # 1. Compute pairwise differences in coordinates
        radial_coords = coords.unsqueeze(dim=2) - coords.unsqueeze(
            dim=1
        )  # [B, N, N, 2] - Relative coordinates between all pairs of tokens

        # 2. Compute L2 norm of radial coordinates
        radial_coord_norm = torch.linalg.norm(radial_coords, ord=2, dim=-1).reshape(
            B * N, N, 1
        )  # [B*N, N, 1] - Euclidean distance

        # Reshape radial_coords for Frame Averaging
        # Equivalent to rearrange(radial_coords, "b n m d -> (b n) m d")
        _, M, D = radial_coords.shape[1:]  # N, 2
        radial_coords = radial_coords.reshape(B * N, M, D)

        # Prepare neighbor masks for Frame Averaging
        if pad_mask is not None:
            # Equivalent to ~rearrange(pad_mask, "b n m -> (b n) m")
            B_mask, N_mask, M_mask = pad_mask.shape
            neighbor_masks = ~(pad_mask.reshape(B_mask * N_mask, M_mask))
        else:
            neighbor_masks = None

        # Create frames and extract features (e.g., orientation, distance)
        frame_feats, _, _ = self.create_frame(
            radial_coords, neighbor_masks
        )  # [B*N*4, N, 2] (for dim=2, n_frames=4)
        frame_feats = frame_feats.view(
            B * N, self.n_frames, N, -1
        )  # [B*N, N_frames, N, Dim]

        # Expand radial_coord_norm to match frame_feats for concatenation
        radial_coord_norm = radial_coord_norm.unsqueeze(dim=1).expand(
            B * N, self.n_frames, N, -1
        )  # [B*N, N_frames, N, 1]

        # Concatenate frame features and radial norm, then compute spatial bias
        # The mean over frames aims to achieve rotation invariance
        spatial_bias = self.edge_bias(
            torch.cat([frame_feats, radial_coord_norm], dim=-1)
        ).mean(dim=1)  # [B * N, N, n_heads]

        # Reshape spatial_bias to [B, H, N, N] to be added to attention scores
        # Equivalent to rearrange(spatial_bias, "(b n) m h -> b h n m", b=B, n=N)
        H_bias = spatial_bias.shape[-1]
        spatial_bias = spatial_bias.view(B, N, M, H_bias)
        spatial_bias = spatial_bias.permute(0, 3, 1, 2)

        """Add spatial bias to attention scores"""
        attn = attn + spatial_bias  # [B, H, N, N]
        if pad_mask is not None:
            # Apply padding mask to attention scores (set to a very small number for softmax)
            pad_mask = pad_mask.unsqueeze(1)  # Expand mask to [B, 1, N, N]
            attn.masked_fill_(pad_mask, -1e9)
        attn = attn.softmax(dim=-1)  # Apply softmax to get attention weights
        attn = self.attn_dropout(attn)  # Apply dropout

        x = attn @ v  # Compute weighted sum of values [B, H, N, D_head]

        # Reshape output back to original D_model dimension
        x = x.transpose(1, 2).reshape(B, N, C)  # [B, N, D_model]
        return self.W_output(x)  # Apply final linear projection and dropout


def get_activation(activation: str = "gelu"):
    """
    Returns the PyTorch activation function module based on its name.

    Args:
        activation (str): Name of the activation function ("gelu", "silu", "relu").

    Returns:
        nn.Module: The corresponding PyTorch activation module.
    """
    return {
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "relu": nn.ReLU,
    }[activation]


class MLP(nn.Module):
    """
    A standard Multi-Layer Perceptron (MLP) block with two linear layers,
    an activation function, and dropout.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        norm_layer: Optional[nn.Module] = None,
        bias: bool = True,
        drop: float = 0.0,
        drop_last: bool = True,
    ):
        """
        Initializes the MLP.

        Args:
            in_features (int): Number of input features.
            hidden_features (Optional[int]): Number of hidden features. Defaults to in_features.
            out_features (Optional[int]): Number of output features. Defaults to in_features.
            act_layer (nn.Module): Activation layer to use.
            norm_layer (Optional[nn.Module]): Normalization layer to use after the first linear layer.
            bias (bool): Whether to use bias in linear layers.
            drop (float): Dropout rate.
            drop_last (bool): Whether to apply dropout after the last linear layer.
        """
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

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for the MLP.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SwiGLUMLP(nn.Module):
    """
    MLP w/ GLU (Gated Linear Unit) style gating using Swish (SiLU) activation.
    This architecture is known for its effectiveness in transformer models.
    See: https://arxiv.org/abs/1612.08083, https://arxiv.org/abs/2002.05202
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        norm_layer: Optional[nn.Module] = None,
        bias: bool = True,
        drop: float = 0.0,
        drop_last: bool = True,
    ):
        """
        Initializes the SwiGLUMLP.

        Args:
            in_features (int): Number of input features.
            hidden_features (Optional[int]): Number of hidden features. Must be even. Defaults to in_features.
            out_features (Optional[int]): Number of output features. Defaults to in_features.
            norm_layer (Optional[nn.Module]): Normalization layer. Applied after the gated activation.
            bias (bool): Whether to use bias in linear layers.
            drop (float): Dropout rate.
            drop_last (bool): Whether to apply dropout after the last linear layer.
        """
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        assert hidden_features % 2 == 0, (
            "hidden_features must be an even number for GLU"
        )

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = nn.SiLU()  # Swish activation
        self.drop1 = nn.Dropout(drop)
        self.norm = (
            norm_layer(hidden_features // 2)
            if norm_layer is not None
            else nn.Identity()
        )
        self.fc2 = nn.Linear(hidden_features // 2, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop) if drop_last else nn.Identity()

    def init_weights(self):
        """
        Initializes weights specifically for the GLU gate, setting the gate portion
        of the first linear layer's bias near zero and the weight near one.
        """
        # override init of fc1 w/ gate portion set to weight near zero, bias=1
        fc1_mid = self.fc1.bias.shape[0] // 2
        nn.init.ones_(self.fc1.bias[fc1_mid:])
        nn.init.normal_(self.fc1.weight[fc1_mid:], std=1e-6)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for the SwiGLUMLP.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        x = self.fc1(x)
        x1, x2 = x.chunk(2, dim=-1)  # Split into two halves for gating
        x = self.act(x1) * x2  # Apply SiLU activation to x1 and multiply by x2 (gating)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


def MLPWrapper(
    in_features: int,
    hidden_features: int,
    out_features: int,
    activation: str = "gelu",
    norm_layer: Optional[nn.Module] = None,
    bias: bool = True,
    drop: float = 0.0,
    drop_last: bool = True,
) -> nn.Module:
    """
    A factory function to create an MLP module, choosing between a standard MLP
    and a SwiGLUMLP based on the 'activation' parameter.

    Args:
        in_features (int): Number of input features.
        hidden_features (int): Number of hidden features.
        out_features (int): Number of output features.
        activation (str): Activation function to use ("gelu", "silu", "relu", "swiglu").
        norm_layer (Optional[nn.Module]): Normalization layer.
        bias (bool): Whether to use bias in linear layers.
        drop (float): Dropout rate.
        drop_last (bool): Whether to apply dropout after the last linear layer.

    Returns:
        nn.Module: An instance of MLP or SwiGLUMLP.
    """
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
    """
    Encodes various input modalities (image features, gene expression,
    technology, and organ information) into a unified embedding space.
    """

    def __init__(self, config: dict):
        """
        Initializes the InputEncoder.

        Args:
            config (dict): Configuration dictionary containing model dimensions
                           like 'd_input', 'd_model', 'n_genes', 'n_tech', 'n_organs'.
        """
        super().__init__()
        self.image_embed = nn.Linear(config["d_input"], config["d_model"])
        self.gene_embed = nn.Linear(config["n_genes"], config["d_model"], bias=False)
        self.tech_embed = nn.Embedding(config["n_tech"], config["d_model"])
        self.organ_embed = nn.Embedding(config["n_organs"], config["d_model"])

    def forward(
        self,
        img_tokens: torch.Tensor,
        ge_tokens: torch.Tensor,
        tech_tokens: Optional[torch.Tensor],
        organ_tokens: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward pass for the InputEncoder.

        Args:
            img_tokens (torch.Tensor): Image feature tokens [Batch, D_input].
            ge_tokens (torch.Tensor): Gene expression tokens [Batch, N_genes].
            tech_tokens (Optional[torch.Tensor]): Technology ID tokens [Batch].
                                                   Can be None if not used.
            organ_tokens (Optional[torch.Tensor]): Organ ID tokens [Batch].
                                                    Can be None if not used.

        Returns:
            torch.Tensor: Combined input embeddings [Batch, D_model].
        """
        # Mandatory embeddings
        img_embed = self.image_embed(img_tokens)
        ge_embed = self.gene_embed(ge_tokens)

        # Accumulate base embeddings
        x = img_embed + ge_embed

        # Conditional embedding for tech_tokens
        if tech_tokens is not None:
            x = x + self.tech_embed(tech_tokens)

        # Conditional embedding for organ_tokens
        if organ_tokens is not None:
            x = x + self.organ_embed(organ_tokens)

        return x


class SpatialTransformer(nn.Module):
    """
    A stack of TransformerBlocks that processes spatial features,
    applying attention considering spatial coordinates.
    """

    def __init__(self, config: dict):
        """
        Initializes the SpatialTransformer.

        Args:
            config (dict): Configuration dictionary containing parameters for
                           TransformerBlocks like 'd_model', 'n_heads', 'n_layers', etc.
        """
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

    def forward(
        self, features: torch.Tensor, coords: torch.Tensor, batch_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for the SpatialTransformer.

        Args:
            features (torch.Tensor): Input features (embeddings) for each token [N_cells, D_model].
            coords (torch.Tensor): Spatial coordinates for each token [N_cells, 2].
            batch_idx (torch.Tensor): Batch index for each cell [N_cells].

        Returns:
            torch.Tensor: Transformed features after passing through transformer blocks [N_cells, D_model].
        """
        # Create a batch-aware padding mask.
        # This mask ensures that attention is not computed between tokens from different batches.
        # It expands batch_idx to create a pairwise comparison matrix:
        # (batch_idx[i] == batch_idx[j]) indicates tokens are in the same batch.
        # We want to mask out (set to True) elements where tokens are NOT in the same batch,
        # hence the negation `~`.
        batch_mask = ~(batch_idx.unsqueeze(0) == batch_idx.unsqueeze(1))

        # Add batch dimension (B=1) for transformer block compatibility
        features = features.unsqueeze(0)  # [1, N_cells, d_model]
        coords = coords.unsqueeze(0)  # [1, N_cells, 2]
        batch_mask = batch_mask.unsqueeze(0)  # [1, N_cells, N_cells]

        for blk in self.blks:
            features = blk(features, coords, padding_mask=batch_mask)
        return features.squeeze(0)  # Remove batch dimension [N_cells, D_model]


class STFM(nn.Module):
    """
    Spatial Transcriptomics Foundation Model (STFM) which combines an InputEncoder
    and a SpatialTransformer to predict gene expression.
    """

    def __init__(self, config: dict = {}) -> None:
        """
        Initializes the STFM.

        Args:
            config (dict): Configuration dictionary. Expected to contain:
                           'd_input', 'd_model', 'n_genes' for encoder and head,
                           and other parameters for SpatialTransformer.
        """
        super(STFM, self).__init__()

        if not config:
            # Default configuration for the STFM model
            self.config = {
                "d_input": 1536,  # Dimension of input image features
                "d_model": 512,  # Dimension of the model's latent space
                "n_layers": 4,  # Number of transformer blocks
                "n_heads": 4,  # Number of attention heads
                "dropout": 0.1,  # Dropout rate for MLP projections
                "attn_dropout": 0.1,  # Dropout rate for attention weights
                "act": "gelu",  # Activation function
                "mlp_ratio": 2.0,  # Ratio for MLP hidden dimension
                "gene_voc_path": hf_hub_download(
                    "RendeiroLab/LazySlide-models", "stpath/symbol2ensembl.json"
                ),
                "model_weigth_path": hf_hub_download(
                    "tlhuang/STPath",
                    "stfm.pth",
                ),
            }
        else:
            self.config = config

        # Initialize all necessary tokenizers
        self.tokenizer = TokenizerTools(
            ge_tokenizer=GeneExpTokenizer(self.config["gene_voc_path"]),
            image_tokenizer=ImageTokenizer(feature_dim=self.config["d_input"]),
            tech_tokenizer=IDTokenizer(id_type="tech"),
            specie_tokenizer=IDTokenizer(id_type="specie"),
            organ_tokenizer=IDTokenizer(id_type="organ"),
            cancer_anno_tokenizer=AnnotationTokenizer(id_type="disease"),
            domain_anno_tokenizer=AnnotationTokenizer(id_type="domain"),
        )

        # Update config with tokenizer-derived dimensions
        self.config["n_genes"] = self.tokenizer.ge_tokenizer.n_tokens
        self.config["n_tech"] = self.tokenizer.tech_tokenizer.n_tokens
        self.config["n_species"] = (
            self.tokenizer.specie_tokenizer.n_tokens
        )  # This key doesn't seem to be used in InputEncoder
        self.config["n_organs"] = self.tokenizer.organ_tokenizer.n_tokens
        self.config["n_cancer_annos"] = (
            self.tokenizer.cancer_anno_tokenizer.n_tokens
        )  # This key doesn't seem to be used in InputEncoder
        self.config["n_domain_annos"] = (
            self.tokenizer.domain_anno_tokenizer.n_tokens
        )  # This key doesn't seem to be used in InputEncoder

        print(self.config)

        self.model = SpatialTransformer(self.config)
        self.input_encoder = InputEncoder(self.config)

        self.gene_exp_head = nn.Sequential(
            nn.LayerNorm(self.config["d_model"]),
            nn.Linear(self.config["d_model"], self.config["n_genes"]),
        )
        if "model_weigth_path" in self.config.keys():
            self.load_state_dict(
                torch.load(self.config["model_weigth_path"], map_location="cpu"),
                strict=True,
            )

    @torch.jit.export
    def inference(
        self,
        img_tokens: torch.Tensor,
        coords: torch.Tensor,
        ge_tokens: torch.Tensor,
        batch_idx: torch.Tensor,
        tech_tokens: Optional[torch.Tensor] = None,
        organ_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Performs the inference pass, encoding inputs and processing them
        through the spatial transformer.

        Args:
            img_tokens (torch.Tensor): Image feature tokens.
            coords (torch.Tensor): Spatial coordinates.
            ge_tokens (torch.Tensor): Gene expression tokens.
            batch_idx (torch.Tensor): Batch index for each cell.
            tech_tokens (Optional[torch.Tensor]): Technology ID tokens.
            organ_tokens (Optional[torch.Tensor]): Organ ID tokens.

        Returns:
            torch.Tensor: Output features from the spatial transformer [N_cells, D_model].
        """
        x = self.input_encoder(
            img_tokens=img_tokens,
            ge_tokens=ge_tokens,
            tech_tokens=tech_tokens,
            organ_tokens=organ_tokens,
        )
        return self.model(x, coords, batch_idx)

    def forward(
        self,
        img_tokens: torch.Tensor,
        coords: torch.Tensor,
        ge_tokens: torch.Tensor,
        batch_idx: torch.Tensor,
        tech_tokens: Optional[torch.Tensor] = None,
        organ_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for the STFM, performing inference and then
        applying the gene expression prediction head.

        Args:
            img_tokens (torch.Tensor): Image feature tokens.
            coords (torch.Tensor): Spatial coordinates.
            ge_tokens (torch.Tensor): Gene expression tokens.
            batch_idx (torch.Tensor): Batch index for each cell.
            tech_tokens (Optional[torch.Tensor]): Technology ID tokens.
            organ_tokens (Optional[torch.Tensor]): Organ ID tokens.

        Returns:
            torch.Tensor: Predicted gene expression values [N_cells, N_genes].
        """
        x = self.inference(
            img_tokens=img_tokens,
            coords=coords,
            batch_idx=batch_idx,
            ge_tokens=ge_tokens,
            tech_tokens=tech_tokens,
            organ_tokens=organ_tokens,
        )
        return self.gene_exp_head(x)


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


@register(
    key="stpath",
    is_gated=False,
    task=ModelTask.feature_prediction,
    license="CC BY-NC-ND 4.0",
    description="A generative foundation model for integrating spatial transcriptomics and whole-slide images",
    commercial=False,
    hf_url="https://huggingface.co/tlhuang/STPath",
    github_url="https://github.com/Graph-and-Geometric-Learning/STPath",
    paper_url="https://doi.org/10.1038/s41746-025-02020-3",
    bib_key="Huang2025-st",
    param_size="~50M",
    encode_dim=512,
    flops=305904607008,
)
class STPath(ModelBase):
    """
    A wrapper class for the STFM that returns a self-describing AnnData object
    including gene symbols, Ensembl IDs, and spatial metadata.
    """

    def __init__(self, config: dict = {}):
        """Initializes the STPath model and its underlying STFM architecture."""
        self.model = STFM(config=config)
        self.model.eval()

    @torch.inference_mode()
    def predict(
        self,
        image_features: torch.Tensor,
        coords: torch.Tensor,
        gene_exp: Optional[sc.AnnData] = None,
        tech: Optional[str] = None,
        organ: Optional[str] = None,
    ) -> sc.AnnData:
        """
        Predicts gene expression and returns a structured AnnData object.
        """
        device = next(self.model.parameters()).device
        n_cells = image_features.shape[0]

        # 1. Coordinate Handling
        # Save original coordinates and create normalized version for the model
        orig_coords_np = coords.clone().cpu()
        norm_coords = coords.clone().cpu()
        norm_coords = normalize_coords(norm_coords)
        norm_coords_tensor = norm_coords.to(device)

        # 2. Token Preparation
        img_tokens = image_features.to(device)

        if gene_exp is None:
            # Use specialized mask_token from the GeneExpTokenizer
            mask_token = self.model.tokenizer.ge_tokenizer.mask_token.to(device)
            ge_tokens = mask_token.unsqueeze(0).expand(n_cells, -1)
        else:
            # Encode input AnnData using the internal tokenizer
            #  TODO: implement padding if just a subset of spots comes with gene expression
            ge_tokens, _ = self.model.tokenizer.ge_tokenizer.encode(gene_exp)
            ge_tokens = ge_tokens.to(device)

        # 3. Metadata Alignment
        tech_tokens = None
        if tech:
            tech_id = self.model.tokenizer.tech_tokenizer.encode(tech, align_first=True)
            tech_tokens = torch.full(
                (n_cells,), tech_id, dtype=torch.long, device=device
            )

        organ_tokens = None
        if organ:
            organ_id = self.model.tokenizer.organ_tokenizer.encode(
                organ, align_first=True
            )
            organ_tokens = torch.full(
                (n_cells,), organ_id, dtype=torch.long, device=device
            )

        # 4. Model Inference
        batch_idx = torch.zeros(n_cells, dtype=torch.long, device=device)
        output = self.model(
            img_tokens=img_tokens,
            coords=norm_coords_tensor,
            ge_tokens=ge_tokens,
            batch_idx=batch_idx,
            tech_tokens=tech_tokens,
            organ_tokens=organ_tokens,
        )

        # 5. Build Metadata Tables
        # Retrieve genes ordered by token ID (Tokens 0 and 1 are special)
        gene_ids = self.model.tokenizer.ge_tokenizer.get_available_genes()

        # Build symbol mapping from the internal dictionary
        # We need to find the symbol for each gene ID stored in the tokenizer
        id_to_symbol = {
            v: k for k, v in self.model.tokenizer.ge_tokenizer.symbol2gene.items()
        }
        gene_symbols = [id_to_symbol.get(gid, "Unknown") for gid in gene_ids]

        # Prepend special tokens to maintain index alignment with output
        all_symbols = ["<pad>", "<mask>"] + gene_symbols
        all_ensembl = ["<pad>", "<mask>"] + gene_ids

        # 6. Final AnnData Construction
        full_adata = sc.AnnData(
            X=output.cpu().numpy(),
            obs={
                "technology": tech if tech else "unknown",
                "organ": organ if organ else "unknown",
            },
            var={"ensembl_id": all_ensembl, "gene_symbol": all_symbols},
            obsm={
                "spatial": norm_coords.cpu().numpy(),
                "spatial_original": orig_coords_np.cpu().numpy(),
            },
        )
        full_adata.var_names = all_symbols

        # Remove the special tokens to return only biological data
        is_real_gene = ~full_adata.var_names.isin(["<pad>", "<mask>"])
        adata = full_adata[:, is_real_gene].copy()

        return adata
