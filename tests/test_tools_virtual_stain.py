import numpy as np
import pytest
from mock_models import MockStyleTransferModel

from lazyslide.tools import virtual_stain


def test_virtual_stain_mock(wsi):
    """Test virtual staining with mock ROSIE model."""
    model = MockStyleTransferModel()

    virtual_stain(
        wsi=wsi,
        model=model,
        batch_size=2,
        num_workers=0,
        pbar=False,
    )

    # Check that the virtual staining result was added to WSI
    assert "rosie_prediction" in wsi.images

    # Get the virtual staining result
    rosie_image = wsi.images["rosie_prediction"]

    # Check image properties
    assert rosie_image.dims == ("c", "y", "x")
    assert rosie_image.shape[0] == 50  # 50 channels
    assert rosie_image.dtype == np.uint8

    # Check channel names are present
    assert hasattr(rosie_image, "c")
    channel_names = list(rosie_image.c.values)
    assert len(channel_names) == 50

    # Check transformations exist
    assert "global" in rosie_image.attrs.get("transform", {})

    # Check that values are in expected range (0-255 for uint8)
    assert rosie_image.values.min() >= 0
    assert rosie_image.values.max() <= 255


def test_virtual_stain_unsupported_model(wsi):
    """Test virtual staining with unsupported model raises error."""
    with pytest.raises((KeyError, ValueError)):
        virtual_stain(
            wsi=wsi,
            model="unsupported_model",
            batch_size=2,
            num_workers=0,
            pbar=False,
        )
