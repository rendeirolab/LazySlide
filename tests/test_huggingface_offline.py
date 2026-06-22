from lazyslide.datasets import _sample


def test_load_dataset_does_not_query_refs_for_release_in_offline_mode(monkeypatch):
    calls = []

    class FailApi:
        def __init__(self):
            raise AssertionError("HfApi should not be used in offline mode")

    def fake_download(repo_id, filename, repo_type, revision, local_files_only=False):
        calls.append(
            {
                "repo_id": repo_id,
                "filename": filename,
                "repo_type": repo_type,
                "revision": revision,
                "local_files_only": local_files_only,
            }
        )
        return f"/tmp/{filename}"

    def fake_open_wsi(slide, store=None):
        return {"slide": slide, "store": store}

    monkeypatch.delenv("LAZYSLIDE_DATASET_REVISION", raising=False)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(_sample, "HfApi", FailApi)
    monkeypatch.setattr(_sample, "hf_hub_download", fake_download)
    monkeypatch.setattr(_sample, "open_wsi", fake_open_wsi)
    monkeypatch.setattr("lazyslide.__version__", "0.3.0")

    wsi = _sample._load_dataset("sample.svs", "sample.zarr.zip", with_data=False)

    assert wsi == {"slide": "/tmp/sample.svs", "store": None}
    assert len(calls) == 1
    assert calls[0]["local_files_only"] is True
    assert calls[0]["revision"] == "v0.3.0"


def test_load_dataset_uses_default_revision_for_dev_versions_offline(monkeypatch):
    calls = []

    class FailApi:
        def __init__(self):
            raise AssertionError("HfApi should not be used in offline mode")

    def fake_download(repo_id, filename, repo_type, revision, local_files_only=False):
        calls.append(
            {
                "repo_id": repo_id,
                "filename": filename,
                "repo_type": repo_type,
                "revision": revision,
                "local_files_only": local_files_only,
            }
        )
        return f"/tmp/{filename}"

    def fake_open_wsi(slide, store=None):
        return {"slide": slide, "store": store}

    monkeypatch.delenv("LAZYSLIDE_DATASET_REVISION", raising=False)
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setattr(_sample, "HfApi", FailApi)
    monkeypatch.setattr(_sample, "hf_hub_download", fake_download)
    monkeypatch.setattr(_sample, "open_wsi", fake_open_wsi)
    monkeypatch.setattr("lazyslide.__version__", "0.3.0.post1")

    wsi = _sample._load_dataset("sample.svs", "sample.zarr.zip", with_data=False)

    assert wsi == {"slide": "/tmp/sample.svs", "store": None}
    assert len(calls) == 1
    assert calls[0]["local_files_only"] is True
    assert calls[0]["revision"] is None


def test_explicit_dataset_revision_bypasses_version_tag_lookup(monkeypatch):
    calls = []

    class FailApi:
        def __init__(self):
            raise AssertionError("HfApi should not be used with an explicit revision")

    def fake_download(repo_id, filename, repo_type, revision, local_files_only=False):
        calls.append(
            {
                "repo_id": repo_id,
                "filename": filename,
                "repo_type": repo_type,
                "revision": revision,
                "local_files_only": local_files_only,
            }
        )
        return f"/tmp/{filename}"

    def fake_open_wsi(slide, store=None):
        return {"slide": slide, "store": store}

    dataset_sha = "d469afd4a763ad366861e8c49d4cf424bfad902c"
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("LAZYSLIDE_DATASET_REVISION", dataset_sha)
    monkeypatch.setattr(_sample, "HfApi", FailApi)
    monkeypatch.setattr(_sample, "hf_hub_download", fake_download)
    monkeypatch.setattr(_sample, "open_wsi", fake_open_wsi)
    monkeypatch.setattr("lazyslide.__version__", "0.12.0")

    wsi = _sample._load_dataset("sample.svs", "sample.zarr.zip", with_data=False)

    assert wsi == {"slide": "/tmp/sample.svs", "store": None}
    assert len(calls) == 1
    assert calls[0]["local_files_only"] is True
    assert calls[0]["revision"] == dataset_sha
