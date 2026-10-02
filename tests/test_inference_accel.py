"""Inference acceleration knobs: amp / autocast_dtype / compile / compile_kws.

See https://github.com/rendeirolab/LazySlide/issues/236
"""

from contextlib import nullcontext

import pytest
import torch

import lazyslide as zs
from lazyslide import _api

from .mock_models import MockImageTextModel


@pytest.fixture(autouse=True)
def reset_settings():
    """Settings is a module-level singleton — restore it after every test."""
    saved = {k: zs.settings[k] for k in zs.settings._attributes}
    yield
    for k, v in saved.items():
        zs.settings[k] = v


class CompileSpy(MockImageTextModel):
    """Records whether (and how) try_compile was called."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.compile_calls = []

    def try_compile(self, **compile_kws):
        self.compile_calls.append(compile_kws)


class NoCompile:
    """A user-supplied object that does not implement try_compile."""


class TestMaybeCompile:
    def test_off_by_default(self):
        spy = CompileSpy()
        _api.maybe_compile(spy)
        assert spy.compile_calls == []

    def test_called_when_requested(self):
        spy = CompileSpy()
        _api.maybe_compile(spy, compile=True, compile_kws={"mode": "max-autotune"})
        assert spy.compile_calls == [{"mode": "max-autotune"}]

    def test_settings_reaches_callers_passing_none(self):
        # Every public function forwards compile=None when the user omits it,
        # so the global switch has to be honored here.
        zs.settings.compile = True
        zs.settings.compile_kws = {"fullgraph": False}
        spy = CompileSpy()
        _api.maybe_compile(spy, compile=None, compile_kws=None)
        assert spy.compile_calls == [{"fullgraph": False}]

    def test_explicit_false_beats_settings(self):
        zs.settings.compile = True
        spy = CompileSpy()
        _api.maybe_compile(spy, compile=False)
        assert spy.compile_calls == []

    def test_no_op_without_try_compile(self):
        model = NoCompile()
        assert _api.maybe_compile(model, compile=True) is model

    def test_returns_the_model(self):
        spy = CompileSpy()
        assert _api.maybe_compile(spy, compile=True) is spy


class TestAutocast:
    def test_off_returns_nullcontext(self):
        assert isinstance(_api.autocast("cpu", amp=False), nullcontext)

    def test_settings_amp(self):
        zs.settings.amp = True
        assert isinstance(_api.autocast("cpu", amp=None), torch.autocast)

    def test_accepts_torch_device(self):
        # torch.autocast wants a device *type* string; passing a torch.device
        # straight through used to raise.
        ctx = _api.autocast(torch.device("cpu"), amp=True)
        with ctx:
            pass

    def test_uses_settings_dtype(self):
        zs.settings.autocast_dtype = torch.bfloat16
        ctx = _api.autocast("cpu", amp=True)
        assert ctx.fast_dtype == torch.bfloat16


class TestLoaderKws:
    def test_no_prefetch_without_workers(self):
        # torch raises when prefetch_factor is set and num_workers == 0
        kws = _api.loader_kws("cpu", num_workers=0, prefetch_factor=4)
        assert "prefetch_factor" not in kws

    def test_prefetch_with_workers(self):
        kws = _api.loader_kws("cpu", num_workers=2, prefetch_factor=4)
        assert kws["prefetch_factor"] == 4

    def test_pin_memory_is_cuda_only(self):
        assert _api.loader_kws("cpu")["pin_memory"] is False
        assert _api.loader_kws("cuda")["pin_memory"] is True
        assert _api.loader_kws(torch.device("cuda:0"))["pin_memory"] is True

    def test_accepted_by_dataloader(self):
        from torch.utils.data import DataLoader, TensorDataset

        ds = TensorDataset(torch.zeros(4, 2))
        DataLoader(ds, batch_size=2, **_api.loader_kws("cpu", num_workers=0))


class TestSettings:
    def test_compile_defaults_off(self):
        assert zs.settings.compile is False
        assert zs.settings.compile_kws == {}

    def test_compile_rejects_non_bool(self):
        with pytest.raises(TypeError):
            zs.settings.compile = "yes"

    def test_compile_kws_rejects_non_dict(self):
        with pytest.raises(TypeError):
            zs.settings.compile_kws = ["mode"]

    def test_compile_kws_none_is_empty(self):
        zs.settings.compile_kws = None
        assert zs.settings.compile_kws == {}

    def test_compile_kws_is_not_mutable_through_the_getter(self):
        # The setter copies on write; the getter has to copy on read too, or a
        # caller (or a test restoring a snapshot) can mutate stored settings.
        zs.settings.compile_kws = {"mode": "max-autotune"}
        zs.settings.compile_kws["leaked"] = True
        assert zs.settings.compile_kws == {"mode": "max-autotune"}

    def test_compile_kws_setter_copies(self):
        source = {"mode": "max-autotune"}
        zs.settings.compile_kws = source
        source["leaked"] = True
        assert zs.settings.compile_kws == {"mode": "max-autotune"}

    def test_exposed_as_settings_keys(self):
        zs.settings["compile"] = True
        assert zs.settings["compile"] is True
        assert "compile_kws" in zs.settings._attributes


class TestEndToEnd:
    """The params must survive the trip through a real public function."""

    def test_compile_param_reaches_model(self):
        spy = CompileSpy()
        zs.tl.text_embedding(["a", "b"], model=spy, compile=True)
        assert spy.compile_calls == [{}]

    def test_compile_kws_reach_model(self):
        spy = CompileSpy()
        zs.tl.text_embedding(
            ["a"], model=spy, compile=True, compile_kws={"dynamic": False}
        )
        assert spy.compile_calls == [{"dynamic": False}]

    def test_settings_compile_reaches_model(self):
        zs.settings.compile = True
        spy = CompileSpy()
        zs.tl.text_embedding(["a"], model=spy)
        assert spy.compile_calls == [{}]

    def test_not_compiled_by_default(self):
        spy = CompileSpy()
        zs.tl.text_embedding(["a"], model=spy)
        assert spy.compile_calls == []


class TestSettingsAmpHonored:
    """Regression guards: these signatures used to make default_value dead code."""

    @pytest.mark.parametrize(
        "func,param",
        [
            (zs.tl.feature_aggregation, "amp"),
            (zs.tl.feature_aggregation, "device"),
            (zs.tl.tile_prediction, "pbar"),
            (zs.tl.virtual_stain, "pbar"),
            (zs.seg.zero_shot, "pbar"),
        ],
    )
    def test_defaults_to_none(self, func, param):
        import inspect

        assert inspect.signature(func).parameters[param].default is None

    def test_text_embedding_stays_on_cpu(self):
        # Deliberate: embedding a few short strings is not worth a device
        # transfer, so this one opts out of settings.device.
        import inspect

        assert inspect.signature(zs.tl.text_embedding).parameters["device"].default == (
            "cpu"
        )

        zs.settings.device = "meta"  # would blow up if it were honored here
        spy = CompileSpy()
        zs.tl.text_embedding(["a"], model=spy)

    def test_cell_runner_reads_settings(self):
        # CellSegmentationRunner used to bypass default_value entirely, so
        # settings.amp worked for seg.tissue/seg.artifact but not seg.cells.
        from lazyslide.segmentation._seg_runner import CellSegmentationRunner

        zs.settings.amp = True
        zs.settings.autocast_dtype = torch.bfloat16

        model = CompileSpy()
        runner = CellSegmentationRunner(
            _StubWSI(), model, transform=lambda x: x, compile=True
        )
        assert runner.amp is True
        assert runner.autocast_dtype is torch.bfloat16
        assert runner.device == zs.settings.device
        assert model.compile_calls == [{}]


class _StubTileSpec:
    base_downsample = 1.0
    overlap_x = 0
    overlap_y = 0


class _StubWSI:
    """Just enough surface for CellSegmentationRunner.__init__."""

    def tile_spec(self, tile_key):
        return _StubTileSpec()
