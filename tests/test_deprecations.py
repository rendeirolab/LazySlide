"""Deprecations, renamed keywords and signature compatibility with v0.12.0."""

import inspect
import re
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from anndata import AnnData

import lazyslide as zs
from lazyslide._setting import Settings
from lazyslide._utils import deprecated_alias, warn_deprecated
from lazyslide.cv import BinaryMask, ProbabilityMap
from lazyslide.cv.transform import TissueDetectionHE
from lazyslide.datasets import _sample
from lazyslide.segmentation import _cell
from lazyslide.segmentation._seg_runner import SemanticSegmentationRunner

from .mock_models import MockSemanticSegmentationModel

# Every deprecation message names the release that removes it
POLICY = r"deprecated since v\d+\.\d+\.\d+ and will be removed in v0\.14\.0"


def _policy(name):
    return re.escape(f"`{name}`") + ".*" + POLICY


class TestHelpers:
    def test_warn_deprecated_blames_the_caller(self):
        with pytest.warns(FutureWarning, match="old") as record:
            warn_deprecated("old")
        assert record[0].filename == __file__

    def test_alias_passes_through_when_unused(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert deprecated_alias("old", None, "new", 1) == 1

    def test_alias_warns_and_maps(self):
        with pytest.warns(FutureWarning, match=_policy("old")):
            assert deprecated_alias("old", 2, "new", None) == 2

    def test_alias_rejects_both(self):
        with pytest.raises(TypeError, match="pass only `new`"):
            deprecated_alias("old", 2, "new", 3)

    def test_alias_default_counts_as_unset(self):
        with pytest.warns(FutureWarning):
            assert deprecated_alias("old", "b", "new", "a", default="a") == "b"


class TestExistingDeprecations:
    def test_artifact_has_no_varargs(self):
        params = inspect.signature(zs.seg.artifact).parameters.values()
        assert all(p.kind is not p.VAR_POSITIONAL for p in params)

    @pytest.mark.parametrize(
        "loader", ["sample", "gtex_artery", "gtex_small_intestine", "lung_carcinoma"]
    )
    def test_datasets_pbar(self, monkeypatch, loader):
        monkeypatch.setattr(_sample, "_dataset_revision", lambda repo_id: None)
        monkeypatch.setattr(_sample, "_download_dataset_file", lambda *a, **k: "slide")
        monkeypatch.setattr(_sample, "open_wsi", lambda *a, **k: "wsi")
        load = getattr(zs.datasets, loader)
        # An explicit False warns too: the parameter itself is going away
        with pytest.warns(FutureWarning, match=_policy("pbar")):
            assert load(with_data=False, pbar=False) == "wsi"
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            load(with_data=False)

    def test_cell_types(self, monkeypatch):
        monkeypatch.setattr(_cell, "cells", lambda *a, **k: None)
        with pytest.warns(FutureWarning, match=_policy("zs.seg.cell_types")):
            zs.seg.cell_types(None)

    def test_models_not_in_star_import(self):
        assert "models" not in zs.__all__
        code = (
            "import inspect, sys, lazyslide\n"
            "from lazyslide import *\n"
            "inspect.getmembers(lazyslide)\n"
            "print('lazyslide.models' in sys.modules)\n"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        assert out.stdout.strip() == "False"


class TestNoOpParams:
    def test_settings_pbar_impl(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            s = Settings()  # constructing must not warn
        assert "pbar_impl" not in s._attributes
        assert s.pbar_impl == "rich"
        with pytest.warns(FutureWarning, match=_policy("settings.pbar_impl")):
            s.pbar_impl = "tqdm"
        with pytest.warns(FutureWarning, match=_policy("settings.pbar_impl")):
            s["pbar_impl"] = "rich"

    def test_rna_linker_gene_name(self):
        pytest.importorskip("scanpy")
        a = AnnData(np.zeros((3, 2), dtype=np.float32))
        b = AnnData(np.zeros((3, 2), dtype=np.float32))
        with pytest.warns(FutureWarning, match=_policy("gene_name")):
            zs.tl.RNALinker(a, b, gene_name="symbol")

    def test_tissue_detection_he(self):
        with pytest.warns(FutureWarning, match=_policy("TissueDetectionHE")):
            TissueDetectionHE()

    def test_probability_map_prob_map(self):
        prob = np.random.default_rng(0).random((8, 8)).astype(np.float32)
        with pytest.warns(FutureWarning, match=_policy("prob_map")):
            ProbabilityMap(prob, prob_map=prob)

    def test_binary_mask_ignore_index(self):
        with pytest.warns(FutureWarning, match=_policy("ignore_index")):
            BinaryMask(np.eye(8, dtype=np.uint8)).to_polygons(ignore_index=0)


class TestWithSlide:
    def test_background_filter_mode(self, wsi_small):
        with pytest.warns(FutureWarning, match=_policy("background_filter_mode")):
            zs.pp.tile_tissues(
                wsi_small, 256, background_filter_mode="exact", key_added="dep_bfm"
            )

    def test_semantic_low_memory(self, wsi_small):
        # An explicit False warns too: the parameter itself is going away
        with pytest.warns(FutureWarning, match=_policy("low_memory")):
            SemanticSegmentationRunner(
                wsi_small, MockSemanticSegmentationModel(), "tiles", low_memory=False
            )

    def test_feature_aggregation_by(self, wsi_small):
        with pytest.warns(FutureWarning, match=_policy("by")):
            zs.tl.feature_aggregation(wsi_small, "resnet50", by="tissue_id")
        assert "agg_tissue_id" in wsi_small["resnet50_tiles"].uns["agg_ops"]
        with pytest.raises(TypeError, match="pass only `agg_by`"):
            zs.tl.feature_aggregation(
                wsi_small, "resnet50", agg_by="tissue_id", by="tissue_id"
            )

    def test_tissue_props_key(self, wsi_small):
        with pytest.warns(FutureWarning, match=_policy("key")):
            zs.tl.tissue_props(wsi_small, key="tissues")
        assert "solidity" in wsi_small["tissues"].columns


def test_spatial_domain_layer():
    # The alias resolves before any work, so a missing slide fails afterwards
    with (
        pytest.warns(FutureWarning, match=_policy("layer")),
        pytest.raises((AttributeError, ImportError)),
    ):
        zs.tl.spatial_domain(None, "features", layer="spatial_features")
    with pytest.raises(TypeError, match="pass only `layer_key`"):
        zs.tl.spatial_domain(None, "features", layer_key="a", layer="b")


def test_zero_shot_show_progress():
    with (
        pytest.warns(FutureWarning, match=_policy("show_progress")),
        pytest.raises(ValueError, match="Prompts"),
    ):
        zs.seg.zero_shot(None, [], "table", "tiles", show_progress=False)


@pytest.mark.parametrize(
    ("func", "old", "new"),
    [
        (zs.tl.feature_aggregation, "by", "agg_by"),
        (zs.tl.spatial_domain, "layer", "layer_key"),
        (zs.tl.tissue_props, "key", "tissue_key"),
        (zs.seg.zero_shot, "show_progress", "pbar"),
    ],
)
def test_renamed_keyword_is_keyword_only(func, old, new):
    params = inspect.signature(func).parameters
    assert params[old].kind is inspect.Parameter.KEYWORD_ONLY
    assert params[old].default is None
    assert params[new].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD


# Positional parameters of v0.12.0, with the renames above applied. Parameters
# added since then must be keyword-only, so that positional calls written
# against v0.12.0 keep binding the same way.
V012_POSITIONAL = [
    (
        zs.seg.cells,
        "wsi model tile_key magnification transform batch_size num_workers device amp autocast_dtype size_filter nucleus_size pbar extract_features low_memory postprocess_workers overlap_ownership key_added",
    ),
    (
        zs.seg.cell_types,
        "wsi model tile_key magnification transform batch_size num_workers device amp autocast_dtype size_filter nucleus_size pbar extract_features low_memory postprocess_workers overlap_ownership key_added",
    ),
    (
        zs.seg.semantic,
        "wsi model tile_key class_names transform mode sigma_scale low_memory threshold ignore_index buffer_px chunk_size batch_size num_workers device amp autocast_dtype pbar key_added",
    ),
    (
        zs.seg.artifact,
        "wsi tile_key model variant mode sigma_scale low_memory threshold buffer_px batch_size num_workers device amp autocast_dtype key_added pbar",
    ),
    (
        zs.seg.SemanticSegmentationRunner,
        "wsi model tile_key transform mode sigma_scale low_memory threshold ignore_index class_names buffer_px chunk_size batch_size num_workers device amp autocast_dtype pbar",
    ),
    (
        zs.seg.CellSegmentationRunner,
        "wsi model tile_key transform size_filter nucleus_size batch_size num_workers device amp autocast_dtype class_names pbar extract_features low_memory postprocess_workers overlap_ownership",
    ),
    (
        zs.seg.zero_shot,
        "wsi prompts table_key tile_key tissue_key threshold model device model_kwargs key_added min_area pbar",
    ),
    (
        zs.tl.tile_prediction,
        "wsi model transform batch_size num_workers tile_key amp autocast_dtype device pbar",
    ),
    (
        zs.tl.virtual_stain,
        "wsi model image_key tile_key device amp autocast_dtype batch_size num_workers pbar",
    ),
    (
        zs.tl.feature_aggregation,
        "wsi feature_key layer_key encoder tile_key agg_by agg_key amp autocast_dtype device",
    ),
    (zs.tl.text_embedding, "texts model amp autocast_dtype device"),
    (
        zs.tl.image_generation,
        "wsi model prompt_tiles tile_key device amp autocast_dtype num_images_per_tiles seed",
    ),
    (zs.tl.spatial_domain, "wsi feature_key tile_key layer_key resolution key_added"),
    (zs.tl.tissue_props, "wsi tissue_key"),
    (
        zs.pp.find_tissues,
        "wsi level refine_level method to_hsv blur_ksize threshold morph_n_iter morph_ksize min_tissue_area min_hole_area detect_holes filter_artifacts disk_radius relaxed_threshold invert_check key_added",
    ),
]


@pytest.mark.parametrize(("func", "names"), V012_POSITIONAL)
def test_positional_parameters_match_v012(func, names):
    params = inspect.signature(func).parameters.values()
    positional = [
        p.name
        for p in params
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.name != "self"
    ]
    assert positional == names.split()


def test_deprecations_follow_policy():
    src = Path(zs.__file__).parent
    for path in src.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "DeprecationWarning" not in text, path
        # Sphinx requires a version on the directive
        assert not re.search(r"\.\. deprecated::\s*$", text, flags=re.MULTILINE), path
