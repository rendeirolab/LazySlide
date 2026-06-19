import geopandas as gpd
import numpy as np
import pytest
from shapely import box
from wsidata import WSIData
from wsidata.io import add_features, add_shapes

from lazyslide.tools import feature_prediction

from .mock_models import MockFeaturePredictionModel


@pytest.fixture
def feature_wsi():
    class Properties:
        def to_dict(self):
            return {}

    class Reader:
        properties = Properties()

    wsi = WSIData(reader=Reader())
    tiles = gpd.GeoDataFrame({"geometry": [box(i, 0, i + 1, 1) for i in range(5)]})
    add_shapes(wsi, "no_spec_tiles", tiles)
    features = np.arange(20, dtype=np.float32).reshape(5, 4)
    add_features(
        wsi,
        key="mock_input_no_spec_tiles",
        tile_key="no_spec_tiles",
        features=features,
    )
    return wsi, features


def test_feature_prediction_batches_without_copying_input(feature_wsi):
    wsi, features = feature_wsi
    model = MockFeaturePredictionModel()

    result = feature_prediction(
        wsi,
        model,
        batch_size=2,
        tile_key="no_spec_tiles",
        pbar=False,
    )

    assert result is None
    result = wsi.tables["mock_feature_prediction_no_spec_tiles"]
    assert result.shape == (5, 2)
    assert list(result.var_names) == ["feature_sum", "feature_mean"]
    np.testing.assert_allclose(result.X[:, 0], features.sum(axis=1))
    np.testing.assert_allclose(result.X[:, 1], features.mean(axis=1))
    assert [len(batch) for batch in model.batches] == [2, 2, 1]
    assert all(np.shares_memory(batch, features) for batch in model.batches)
    assert list(result.obs["tile_id"]) == list(range(5))


def test_feature_prediction_explicit_keys(feature_wsi):
    wsi, _ = feature_wsi
    model = MockFeaturePredictionModel()

    result = feature_prediction(
        wsi,
        model,
        feature_key="mock_input_no_spec_tiles",
        tile_key="no_spec_tiles",
        key_added="custom_predictions",
        pbar=False,
    )

    assert result is None
    assert "custom_predictions" in wsi.tables


def test_feature_prediction_resolves_registered_model(feature_wsi, monkeypatch):
    import lazyslide_models

    wsi, _ = feature_wsi
    monkeypatch.setattr(
        lazyslide_models,
        "MODEL_REGISTRY",
        {"mock-predictor": MockFeaturePredictionModel},
    )

    result = feature_prediction(
        wsi,
        "mock-predictor",
        tile_key="no_spec_tiles",
        pbar=False,
    )

    assert result is None
    assert wsi.tables["mock-predictor_no_spec_tiles"].shape == (5, 2)
    assert "mock-predictor_no_spec_tiles" in wsi.tables


def test_feature_prediction_requires_feature_key(feature_wsi):
    wsi, _ = feature_wsi

    class ModelWithoutFeatureName(MockFeaturePredictionModel):
        features_model_name = None

    with pytest.raises(ValueError, match="feature_key is required"):
        feature_prediction(
            wsi,
            ModelWithoutFeatureName(),
            tile_key="no_spec_tiles",
            pbar=False,
        )


@pytest.mark.parametrize(
    ("output", "error"),
    [
        (np.ones(2), "non-empty mapping"),
        ({"bad": np.ones((2, 2))}, "must have shape"),
        ({"bad": np.ones(1)}, "has 1 rows"),
    ],
)
def test_feature_prediction_validates_model_output(feature_wsi, output, error):
    wsi, _ = feature_wsi

    class BadModel(MockFeaturePredictionModel):
        def predict(self, features):
            return output

    with pytest.raises((TypeError, ValueError), match=error):
        feature_prediction(
            wsi,
            BadModel(),
            feature_key="mock_input_no_spec_tiles",
            tile_key="no_spec_tiles",
            batch_size=2,
            pbar=False,
        )


def test_feature_prediction_rejects_non_protocol_model(feature_wsi):
    wsi, _ = feature_wsi

    class PredictOnlyModel:
        def predict(self, features):
            return {"value": np.asarray(features)[:, 0]}

    with pytest.raises(TypeError, match="FeaturePredictionModelProtocol"):
        feature_prediction(
            wsi,
            PredictOnlyModel(),
            feature_key="mock_input_no_spec_tiles",
            tile_key="no_spec_tiles",
            pbar=False,
        )


def test_feature_prediction_rejects_invalid_batch_size(feature_wsi):
    wsi, _ = feature_wsi
    with pytest.raises(ValueError, match="batch_size"):
        feature_prediction(
            wsi,
            MockFeaturePredictionModel(),
            tile_key="no_spec_tiles",
            batch_size=0,
            pbar=False,
        )
