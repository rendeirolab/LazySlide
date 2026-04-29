from __future__ import annotations

import pytest
import torch
from lazyslide_models import list_models
from lazyslide_models._model_registry import MODEL_REGISTRY

# ── CLI options ───────────────────────────────────────────────────────────────


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device for model tests (default: cpu)",
    )
    parser.addoption(
        "--skip-models",
        default="",
        help="Comma-separated model names to skip (e.g. 'gigapath,sam')",
    )


# ── Session fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def device(request: pytest.FixtureRequest) -> str:
    d = request.config.getoption("--device")
    if d == "cuda" and not torch.cuda.is_available():
        pytest.skip("--device=cuda requested but CUDA is not available")
    if d == "mps" and not torch.backends.mps.is_available():
        pytest.skip("--device=mps requested but MPS is not available")
    return d


@pytest.fixture(scope="session")
def skip_models(request: pytest.FixtureRequest) -> frozenset[str]:
    raw = request.config.getoption("--skip-models")
    return frozenset(n.strip() for n in raw.split(",") if n.strip())


@pytest.fixture(scope="session")
def load_model(device: str, skip_models: frozenset[str]):
    """
    Session-scoped factory fixture.

    Call ``load_model(model_name)`` inside a test to get an initialised,
    device-placed model.  All skip logic (gated, missing deps, not-implemented,
    manual --skip-models) is handled here so test functions stay clean.

    Each model is loaded once and cached for the whole session.
    """
    from huggingface_hub.errors import GatedRepoError

    cache: dict[str, object] = {}

    def _load(name: str):
        if name in cache:
            return cache[name]

        if name in skip_models:
            pytest.skip(f"'{name}' in --skip-models list")

        try:
            model = MODEL_REGISTRY[name]()
        except GatedRepoError:
            pytest.skip(f"'{name}' is gated (no HF credentials present)")
        except ModuleNotFoundError as exc:
            pytest.skip(f"'{name}' missing optional dependency: {exc}")
        except NotImplementedError:
            pytest.skip(f"'{name}' is not implemented yet")

        model.to(device)
        cache[name] = model
        return model

    return _load


# ── Parametrization helper ────────────────────────────────────────────────────


def models_for_task(task: str) -> list[pytest.param]:
    """
    Return a list of ``pytest.param`` objects for every model registered under
    *task*.  Gated models receive the ``gated`` mark so they can be filtered
    with ``-m 'not gated'``.
    """
    params = []
    for name in list_models(task=task):
        meta = MODEL_REGISTRY[name]
        marks = []
        if getattr(meta, "is_gated", False):
            marks.append(pytest.mark.gated)
        params.append(pytest.param(name, marks=marks, id=name))
    return params
