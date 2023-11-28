from functools import partial
from lazyslide.readers.utils import get_crop_left_top_width_height

import pytest

IMAGE_WIDTH = 200
IMAGE_HEIGHT = 100

test_func = partial(get_crop_left_top_width_height,
                    img_width=IMAGE_WIDTH,
                    img_height=IMAGE_HEIGHT)


def test_get_crop_xy_wh():

    # inside
    assert test_func(left=50, top=50, width=10, height=10) == (50, 50, 10, 10, None)
    # upper-left
    assert test_func(left=-10, top=-10, width=20, height=20) == (0, 0, 10, 10, "south-east")
    # center-left
    assert test_func(left=-10, top=50, width=20, height=20) == (0, 50, 10, 20, "east")
    # lower-left
    assert test_func(left=-10, top=90, width=20, height=20) == (0, 90, 10, 10, "north-east")
    # upper-center
    assert test_func(left=50, top=-10, width=20, height=20) == (50, 0, 20, 10, "south")
    # lower-center
    assert test_func(left=50, top=90, width=20, height=20) == (50, 90, 20, 10, "north")
    # upper-right
    assert test_func(left=190, top=-10, width=20, height=20) == (190, 0, 10, 10, "south-west")
    # center-right
    assert test_func(left=190, top=50, width=20, height=20) == (190, 50, 10, 20, "west")
    # lower-right
    assert test_func(left=190, top=90, width=20, height=20) == (190, 90, 10, 10, "north-west")


def test_get_crop_xy_wh_outside():
    with pytest.raises(RuntimeError):
        test_func(left=-5, top=-20, width=10, height=10)

    with pytest.raises(RuntimeError):
        test_func(left=-20, top=-20, width=10, height=10)

    with pytest.raises(RuntimeError):
        test_func(left=-20, top=50, width=10, height=10)

    with pytest.raises(RuntimeError):
        test_func(left=-20, top=120, width=10, height=10)

    with pytest.raises(RuntimeError):
        test_func(left=50, top=-20, width=10, height=10)

