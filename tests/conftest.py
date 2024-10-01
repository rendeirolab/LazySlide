from pathlib import Path

import pytest
import torch


class MockNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.zeros(1000)


@pytest.fixture(scope="session", autouse=True)
def test_slide():
    return Path(__file__).parent / "data" / "CMU-1-Small-Region.svs"


@pytest.fixture(scope="session", autouse=True)
def wsi(test_slide):
    from wsidata import open_wsi

    return open_wsi(test_slide)


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
