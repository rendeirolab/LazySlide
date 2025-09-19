import numpy as np
import pytest

from lazyslide.models.style_transfer import ROSIE
from lazyslide.tools import virtual_stain

# Import get_token to check if user is logged in to HuggingFace
try:
    from huggingface_hub.utils._auth import get_token

    HF_TOKEN_AVAILABLE = get_token() is not None
except ImportError:
    HF_TOKEN_AVAILABLE = False


# Define a function to conditionally skip tests if HF token is not available
def skip_if_no_token(*model_class):
    """Skip test if no HuggingFace token is available for gated models."""
    if not HF_TOKEN_AVAILABLE:
        return pytest.param(
            *model_class, marks=pytest.mark.skip(reason="Requires HF token")
        )
    return pytest.param(*model_class)


def test_virtual_stain_rosie(wsi):
    """Test virtual staining with ROSIE model."""
    # Skip if token required
    skip_if_no_token(ROSIE)

    # Test virtual staining function
    virtual_stain(
        wsi=wsi,
        model="rosie",
        batch_size=2,  # Small batch size for testing
        num_workers=0,  # No multiprocessing for tests
        pbar=False,  # Disable progress bar for cleaner test output
    )

    # Check that the virtual staining result was added to WSI
    assert "rosie_prediction" in wsi.images

    # Get the virtual staining result
    rosie_image = wsi.images["rosie_prediction"]

    # Check image properties
    assert rosie_image.dims == ("c", "y", "x")  # Channel, Y, X dimensions
    assert rosie_image.shape[0] == 50  # 50 channels for ROSIE
    assert rosie_image.dtype == np.uint8  # Should be uint8 after processing

    # Check channel names are present
    assert hasattr(rosie_image, "c")
    channel_names = list(rosie_image.c.values)
    assert len(channel_names) == 50

    # Check transformations exist
    assert "global" in rosie_image.attrs.get("transform", {})


def test_virtual_stain_unsupported_model(wsi):
    """Test virtual staining with unsupported model raises error."""
    with pytest.raises(ValueError, match="Model unsupported_model not supported"):
        virtual_stain(
            wsi=wsi,
            model="unsupported_model",
            batch_size=2,
            num_workers=0,
            pbar=False,
        )


def test_virtual_stain_postprocessing(wsi):
    """Test that virtual staining applies proper postprocessing."""
    # Skip if token required
    skip_if_no_token(ROSIE)

    # Run virtual staining
    virtual_stain(
        wsi=wsi,
        model="rosie",
        batch_size=2,
        num_workers=0,
        pbar=False,
    )

    # Get the result
    rosie_image = wsi.images["rosie_prediction"]

    # Check that values are in expected range (0-255 for uint8)
    assert rosie_image.values.min() >= 0
    assert rosie_image.values.max() <= 255

    # Check data type is uint8 (result of postprocessing)
    assert rosie_image.dtype == np.uint8

    # Check that image has been processed (not all zeros)
    assert not np.all(rosie_image.values == 0)
