from functools import partial
from lazyslide.readers.utils import get_crop_xy_wh

import pytest

IMAGE_WIDTH = 100
IMAGE_HEIGHT = 100

test_func = partial(get_crop_xy_wh,
                    img_width=IMAGE_WIDTH,
                    img_height=IMAGE_HEIGHT)


def test_get_crop_xy_wh():

    # inside
    assert test_func(x=50, y=50, width=10, height=10) == (50, 50, 10, 10, None)
    # upper-left
    assert test_func(x=-10, y=-10, width=20, height=20) == (0, 0, 10, 10, "south-east")
    # center-left
    assert test_func(x=-10, y=50, width=20, height=20) == (0, 50, 10, 20, "east")
    # lower-left
    assert test_func(x=-10, y=90, width=20, height=20) == (0, 90, 10, 10, "north-east")
    # upper-center
    assert test_func(x=50, y=-10, width=20, height=20) == (50, 0, 20, 10, "south")
    # lower-center
    assert test_func(x=50, y=90, width=20, height=20) == (50, 90, 20, 10, "north")
    # upper-right
    assert test_func(x=90, y=-10, width=20, height=20) == (90, 0, 10, 10, "south-west")
    # center-right
    assert test_func(x=90, y=50, width=20, height=20) == (90, 50, 10, 20, "west")
    # lower-right
    assert test_func(x=90, y=90, width=20, height=20) == (90, 90, 10, 10, "north-west")


def test_get_crop_xy_wh_outside():
    with pytest.raises(RuntimeError):
        test_func(x=-5, y=-20, width=10, height=10)

    with pytest.raises(RuntimeError):
        test_func(x=-20, y=-20, width=10, height=10)

    with pytest.raises(RuntimeError):
        test_func(x=-20, y=50, width=10, height=10)

    with pytest.raises(RuntimeError):
        test_func(x=-20, y=120, width=10, height=10)

    with pytest.raises(RuntimeError):
        test_func(x=50, y=-20, width=10, height=10)

