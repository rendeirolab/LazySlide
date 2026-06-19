from pathlib import Path
from types import SimpleNamespace
from xml.etree.ElementTree import ParseError

import pytest
from shapely.geometry import LineString, Point, Polygon

from lazyslide.io import _annotation as annotation_module
from lazyslide.io import load_annotations, read_ndpa

DATA_DIR = Path(__file__).parent / "data"


class StubWSI:
    def __init__(self, raw_properties=None, mpp=0.75):
        self.properties = SimpleNamespace(shape=(1000, 2000), mpp=mpp)
        self.raw_properties = raw_properties or {
            "openslide.mpp-x": "0.5",
            "openslide.mpp-y": "1.0",
            "hamamatsu.XOffsetFromSlideCentre": "100000",
            "hamamatsu.YOffsetFromSlideCentre": "-200000",
        }

    def __contains__(self, key):
        return False


def test_read_ndpa_mixed_geometries_and_metadata():
    annotations = read_ndpa(StubWSI(), DATA_DIR / "annotations_mixed.ndpa")

    assert annotations.annotation_type.tolist() == ["freehand", "freehand", "pin"]
    assert annotations.title.tolist() == ["Tumor", "Margin", "Review here"]
    assert isinstance(annotations.geometry.iloc[0], Polygon)
    assert isinstance(annotations.geometry.iloc[1], LineString)
    assert isinstance(annotations.geometry.iloc[2], Point)
    assert annotations.geometry.iloc[0].bounds == pytest.approx((100, 200, 300, 400))
    assert list(annotations.geometry.iloc[1].coords) == pytest.approx(
        [(500, 600), (700, 800)]
    )
    assert annotations.geometry.iloc[2].coords[0] == pytest.approx((900, 100))


def test_load_annotations_dispatches_ndpa_and_keeps_non_area_geometries(monkeypatch):
    added = {}

    def capture_shapes(wsi, key, annotations):
        added[key] = annotations

    monkeypatch.setattr(annotation_module, "add_shapes", capture_shapes)

    load_annotations(
        StubWSI(),
        DATA_DIR / "annotations_mixed.ndpa",
        min_area=10_000,
        key_added="ndpa",
    )

    annotations = added["ndpa"]
    assert annotations.geometry.geom_type.tolist() == ["Polygon", "LineString", "Point"]


def test_read_ndpa_uses_scalar_mpp_as_fallback():
    raw_properties = {
        "hamamatsu.XOffsetFromSlideCentre": "100000",
        "hamamatsu.YOffsetFromSlideCentre": "-200000",
    }
    annotations = read_ndpa(
        StubWSI(raw_properties=raw_properties, mpp=0.5),
        DATA_DIR / "annotations_mixed.ndpa",
    )

    assert annotations.geometry.iloc[2].coords[0] == pytest.approx((900, -300))


@pytest.mark.parametrize(
    ("raw_properties", "mpp", "match"),
    [
        (
            {
                "hamamatsu.XOffsetFromSlideCentre": "0",
                "hamamatsu.YOffsetFromSlideCentre": "0",
            },
            None,
            "microns-per-pixel",
        ),
        ({"openslide.mpp-x": "0.5", "openslide.mpp-y": "0.5"}, 0.5, "Offset"),
    ],
)
def test_read_ndpa_requires_coordinate_metadata(raw_properties, mpp, match):
    with pytest.raises(ValueError, match=match):
        read_ndpa(
            StubWSI(raw_properties=raw_properties, mpp=mpp),
            DATA_DIR / "annotations_mixed.ndpa",
        )


def test_read_ndpa_rejects_malformed_xml():
    with pytest.raises(ParseError):
        read_ndpa(StubWSI(), DATA_DIR / "annotations_malformed.ndpa")
