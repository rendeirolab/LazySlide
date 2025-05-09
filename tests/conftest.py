import os

import pytest
import torch


class MockNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.zeros(x.shape[0], 1000)


@pytest.fixture(scope="session", autouse=True)
def wsi():
    import lazyslide as zs

    return zs.datasets.gtex_artery()


@pytest.fixture(scope="session")
def tmp_path_session(tmp_path_factory):
    return tmp_path_factory.mktemp("session_tmp")


@pytest.fixture(scope="session", autouse=True)
def torch_model_file(tmp_path_session):
    model = MockNet()
    torch.save(model, tmp_path_session / "model.pt")
    return tmp_path_session / "model.pt"


@pytest.fixture(scope="session", autouse=True)
def torch_jit_file(tmp_path_session):
    model = MockNet()
    torch.jit.script(model).save(tmp_path_session / "jit_model.pt")
    return tmp_path_session / "jit_model.pt"


def pytest_collection_modifyitems(config, items):
    if os.getenv("GITHUB_ACTIONS") == "true":
        skip_on_ci = pytest.mark.skip(reason="Skipped on GitHub CI")
        for item in items:
            if "skip_on_ci" in item.keywords:
                item.add_marker(skip_on_ci)
