import anndata as ad
import numpy as np
import pandas as pd
import pytest

import lazyslide as zs

pytest.importorskip("scanpy")


def _linker():
    rng = np.random.default_rng(0)
    obs = pd.DataFrame({"g": ["a", "b"] * 10}, index=[f"s{i}" for i in range(20)])
    features = ad.AnnData(rng.normal(size=(20, 10)).astype("float32"), obs=obs)
    features.var_names = [str(i) for i in range(10)]
    omics = ad.AnnData(rng.normal(size=(20, 5)).astype("float32"))
    return zs.tl.RNALinker(features, omics), omics


def test_associate_defaults_to_last_score_key():
    """Regression: associate() looked up obs[None] instead of the last score."""
    linker, omics = _linker()
    linker.score("g", "a", n_features=3)
    linker.associate(method="pearson")
    assert "association_score" in omics.var


def test_associate_before_score_raises():
    linker, _ = _linker()
    with pytest.raises(ValueError, match="score"):
        linker.associate()


def test_associate_uses_an_explicit_score_key():
    """Regression: a falsy score_key such as "" fell back to the last score."""
    linker, _ = _linker()
    linker.score("g", "a", n_features=3)
    with pytest.raises(KeyError):
        linker.associate(score_key="")
