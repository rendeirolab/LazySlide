"""Model parameters take a registry key or a model instance."""

import inspect
import re

import pytest
from lazyslide_models import MODEL_REGISTRY

import lazyslide as zs
from lazyslide.segmentation._zero_shot import _initialize_model

from .mock_models import MockSemanticSegmentationModel


class _Resolved(Exception):
    """Raised by a registry stub to stop a function once its model is built."""


def _stub(monkeypatch, key):
    built = []

    class Stub:
        def __init__(self, **kwargs):
            built.append(key)
            raise _Resolved

    monkeypatch.setitem(MODEL_REGISTRY, key, Stub)
    return built


@pytest.mark.parametrize(
    ("name", "key"),
    [
        ("grandqc", "grandqc-tissue"),
        ("hest", "hest-tissue-segmentation"),
        ("pathprofiler", "pathprofiler"),
        ("grandqc-tissue", "grandqc-tissue"),
    ],
)
def test_tissue_resolves_through_the_registry(monkeypatch, name, key):
    built = _stub(monkeypatch, key)
    with pytest.raises(_Resolved):
        zs.seg.tissue(None, model=name)
    assert built == [key]


class TestWithSlide:
    def test_tissue_takes_an_instance(self, wsi_small):
        # The mock puts p=0.8 on channel 1 everywhere, so the slide is all tissue
        zs.seg.tissue(
            wsi_small, model=MockSemanticSegmentationModel(), mpp=8, key_added="mp_t"
        )
        assert len(wsi_small["mp_t"]) > 0

    def test_tissue_takes_a_single_channel_model(self, wsi_small):
        """Regression: an unlabeled single-channel model indexed channel 1."""
        import torch
        from lazyslide_models.base import SegmentationOutput

        class OneChannel(MockSemanticSegmentationModel):
            def segment(self, image):
                b, _, h, w = image.shape
                return SegmentationOutput(probability_map=torch.full((b, 1, h, w), 0.9))

        zs.seg.tissue(wsi_small, model=OneChannel(), mpp=8, key_added="mp_t1")
        assert len(wsi_small["mp_t1"]) > 0

    def test_tissue_needs_mpp_for_unknown_models(self, wsi_small):
        with pytest.raises(ValueError, match="pass `mpp` or `level`"):
            zs.seg.tissue(wsi_small, model=MockSemanticSegmentationModel())

    def test_artifact_uses_the_model_classes(self, wsi_small):
        class Named(MockSemanticSegmentationModel):
            classes = (
                "Background",
                "Normal Tissue",
                "Crease",
                *(f"c{i}" for i in range(3, 8)),
            )

        # 256 px tiles at 0.5 mpp would fail GrandQC's checks; they're not applied
        zs.seg.artifact(wsi_small, "tiles", model=Named(), key_added="mp_art")
        assert set(wsi_small["mp_art"]["class"]) == {"Crease"}

    def test_semantic_takes_a_registry_key(self, wsi_small, monkeypatch):
        monkeypatch.setitem(
            MODEL_REGISTRY, "mock-semantic", MockSemanticSegmentationModel
        )
        zs.seg.semantic(wsi_small, "mock-semantic", key_added="mp_sem")
        assert len(wsi_small["mp_sem"]) > 0


def test_image_generation_takes_an_instance():
    class Generator:
        def to(self, device):
            return self

        def generate(self, **kwargs):
            return ["image"] * kwargs["num_images_per_prompt"]

    assert (
        zs.tl.image_generation(model=Generator(), num_images_per_tiles=3)
        == ["image"] * 3
    )


@pytest.mark.parametrize(
    ("call", "method"),
    [
        (lambda m: zs.tl.zero_shot_score(None, ["a"], "f", model=m), "score"),
        (lambda m: zs.tl.slide_caption(None, ["a"], "f", model=m), "caption"),
        (lambda m: _initialize_model(m), "get_image_embedding"),
    ],
)
def test_capability_is_checked(call, method):
    with pytest.raises(TypeError, match=f"`{method}`"):
        call(object())


def test_no_closed_model_option_lists():
    """`model` params are typed and documented as a key or an instance."""
    for mod in (zs.pp, zs.tl, zs.seg, zs.pl, zs.io):
        for name in mod.__all__:
            func = getattr(mod, name)
            if not callable(func) or inspect.isclass(func):
                continue
            param = inspect.signature(func).parameters.get("model")
            if param is None:
                continue
            assert "Literal" not in str(param.annotation), name
            doc_type = re.search(r"^\s*model : (.*)$", func.__doc__ or "", re.MULTILINE)
            assert doc_type is None or "{" not in doc_type.group(1), name
