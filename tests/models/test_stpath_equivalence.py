"""
Numerical equivalence test between the LazySlide STFM reimplementation
and the original STPath implementation.

The original STPath repository is cloned from GitHub at test time so the
comparison is always made against upstream.  Set the STPATH_REPO environment
variable to an existing local clone to skip the network step (useful during
development).
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

STPATH_GIT_URL = "https://github.com/Graph-and-Geometric-Learning/STPath.git"

# ---------------------------------------------------------------------------
# Tiny synthetic gene vocabulary
# 5 symbols → 5 unique Ensembl IDs → n_tokens = 5 + 2 (pad + mask) = 7
# ---------------------------------------------------------------------------
MINI_GENE_VOC = {
    "GENE_A": "ENSG00000000001",
    "GENE_B": "ENSG00000000002",
    "GENE_C": "ENSG00000000003",
    "GENE_D": "ENSG00000000004",
    "GENE_E": "ENSG00000000005",
}

# Small architecture so the test runs quickly (no weight download needed)
FEATURE_DIM = 16  # d_input (real model uses 1536 for GigaPath features)
D_MODEL = 16  # must be divisible by N_HEADS
N_LAYERS = 2
N_HEADS = 4
N_SPOTS = 8


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def stpath_repo(tmp_path_factory):
    """
    Return a path to the original STPath repository.

    Prefers a local copy set via the STPATH_REPO environment variable;
    falls back to cloning from GitHub.  The test is skipped when neither
    a local copy nor network access is available.
    """
    local = os.environ.get("STPATH_REPO")
    if local and os.path.isdir(os.path.join(local, "stpath")):
        return local

    clone_dir = str(tmp_path_factory.mktemp("stpath_repo"))
    result = subprocess.run(
        ["git", "clone", "--depth=1", STPATH_GIT_URL, clone_dir],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip(
            f"Could not clone STPath repository (network unavailable?): {result.stderr}"
        )
    return clone_dir


@pytest.fixture(scope="module")
def gene_voc_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("vocab") / "symbol2ensembl.json"
    path.write_text(json.dumps(MINI_GENE_VOC))
    return str(path)


@pytest.fixture(scope="module")
def orig_stfm(stpath_repo, gene_voc_file):
    """Original STFM from the cloned STPath repository with random weights."""
    if stpath_repo not in sys.path:
        sys.path.insert(0, stpath_repo)

    from stpath.model.model import STFM as OrigSTFM
    from stpath.model.nn_utils.config import ModelConfig
    from stpath.tokenization import GeneExpTokenizer, IDTokenizer

    ge_tok = GeneExpTokenizer(gene_voc_file)
    tech_tok = IDTokenizer(id_type="tech")
    organ_tok = IDTokenizer(id_type="organ")

    cfg = ModelConfig(
        d_input=FEATURE_DIM,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        n_genes=ge_tok.n_tokens,
        n_tech=tech_tok.n_tokens,
        n_organs=organ_tok.n_tokens,
        dropout=0.0,
        attn_dropout=0.0,
        act="gelu",
        mlp_ratio=2.0,
        backbone="spatial_transformer",
        activation="gelu",
        feature_dim=FEATURE_DIM,
    )
    torch.manual_seed(0)
    return OrigSTFM(cfg).eval()


@pytest.fixture(scope="module")
def lazy_stfm(gene_voc_file, orig_stfm):
    """LazySlide STFM loaded with the same weights as the original."""
    from lazyslide.models.tile_prediction.stpath import STFM as LazySTFM

    config = {
        "d_input": FEATURE_DIM,
        "d_model": D_MODEL,
        "n_layers": N_LAYERS,
        "n_heads": N_HEADS,
        "dropout": 0.0,
        "attn_dropout": 0.0,
        "act": "gelu",
        "mlp_ratio": 2.0,
        "gene_voc_path": gene_voc_file,
        # No "model_weigth_path" — weights are transferred from orig_stfm below
    }
    model = LazySTFM(config).eval()
    # State-dict keys are identical between both implementations
    model.load_state_dict(orig_stfm.state_dict())
    return model


def _make_inputs(n_genes: int, stpath_repo: str) -> tuple:
    """Return a deterministic set of synthetic inputs."""
    torch.manual_seed(42)

    if stpath_repo not in sys.path:
        sys.path.insert(0, stpath_repo)
    from stpath.tokenization import IDTokenizer

    tech_id = IDTokenizer(id_type="tech").encode("Visium", align_first=True)
    organ_id = IDTokenizer(id_type="organ").encode("Kidney", align_first=True)

    img_tokens = torch.randn(N_SPOTS, FEATURE_DIM)
    coords = torch.rand(N_SPOTS, 2) * 100
    ge_tokens = torch.zeros(N_SPOTS, n_genes)
    ge_tokens[:, 1] = 1.0  # mask token at index 1
    batch_idx = torch.zeros(N_SPOTS, dtype=torch.long)
    tech_tokens = torch.full((N_SPOTS,), tech_id, dtype=torch.long)
    organ_tokens = torch.full((N_SPOTS,), organ_id, dtype=torch.long)

    return img_tokens, coords, ge_tokens, batch_idx, tech_tokens, organ_tokens


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_state_dict_keys_match(orig_stfm, lazy_stfm):
    """Both models must expose the same set of state-dict key names."""
    orig_keys = set(orig_stfm.state_dict().keys())
    lazy_keys = set(lazy_stfm.state_dict().keys())
    assert orig_keys == lazy_keys, (
        f"State-dict key mismatch.\n"
        f"  Only in original : {orig_keys - lazy_keys}\n"
        f"  Only in LazySlide: {lazy_keys - orig_keys}"
    )


def test_numerical_equivalence(stpath_repo, orig_stfm, lazy_stfm):
    """
    Given identical weights and inputs both STFM implementations must produce
    bit-identical outputs (within float32 tolerance).
    """
    n_genes = lazy_stfm.tokenizer.ge_tokenizer.n_tokens
    img_tokens, coords, ge_tokens, batch_idx, tech_tokens, organ_tokens = _make_inputs(
        n_genes, stpath_repo
    )

    with torch.no_grad():
        # Original exposes the full forward pass as prediction_head()
        out_orig = orig_stfm.prediction_head(
            img_tokens, coords, ge_tokens, batch_idx, tech_tokens, organ_tokens
        )
        # LazySlide exposes it via __call__ / forward()
        out_lazy = lazy_stfm(
            img_tokens, coords, ge_tokens, batch_idx, tech_tokens, organ_tokens
        )

    assert out_orig.shape == out_lazy.shape, (
        f"Shape mismatch: original {out_orig.shape} vs LazySlide {out_lazy.shape}"
    )
    np.testing.assert_allclose(
        out_lazy.numpy(),
        out_orig.numpy(),
        atol=1e-5,
        rtol=1e-5,
        err_msg=(
            "LazySlide STFM output differs from the original implementation. "
            "Check FrameAveraging, Attention, or MLP layers for divergence."
        ),
    )
