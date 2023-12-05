import numpy as np
from numba import njit


@njit(cache=True)
def get_crop_left_top_width_height(img_width, img_height,
                                   left, top,
                                   width, height):
    top_in = 0 <= top <= img_height
    left_in = 0 <= left <= img_width
    bottom_in = 0 <= (top + height) <= img_height
    right_in = 0 <= (left + width) <= img_width
    top_out, bottom_out = not top_in, not bottom_in
    left_out, right_out = not left_in, not right_in
    # If extract from region outside image
    if (top_out and bottom_out) or (left_out and right_out):
        raise RuntimeError(f"Extracting region that are completely outside image. \n"
                           f"Image shape: H, W ({img_height}, {img_width}) \n"
                           f"Tile: Top, Left, Width, Height ({top}, {left}, {width}, {height})")

    if top_out and bottom_in and left_out and right_in:
        crop_left, crop_top = 0, 0
        crop_w, crop_h = width + left, height + top
        pos = "south-east"
    elif top_out and bottom_in and left_in and right_in:
        crop_left, crop_top = left, 0
        crop_w, crop_h = width, height + top
        pos = "south"
    elif top_out and bottom_in and left_in and right_out:
        crop_left, crop_top = left, 0
        crop_w, crop_h = img_width - left, height + top
        pos = "south-west"
    elif top_in and bottom_in and left_out and right_in:
        crop_left, crop_top = 0, top
        crop_w, crop_h = width + left, height
        pos = "east"
    elif top_in and bottom_in and left_in and right_out:
        crop_left, crop_top = left, top
        crop_w, crop_h = img_width - left, height
        pos = "west"
    elif top_in and bottom_out and left_out and right_in:
        crop_left, crop_top = 0, top
        crop_w, crop_h = width + left, img_height - top
        pos = "north-east"
    elif top_in and bottom_out and left_in and right_in:
        crop_left, crop_top = left, top
        crop_w, crop_h = width, img_height - top
        pos = "north"
    elif top_in and bottom_out and left_in and right_out:
        crop_left, crop_top = left, top
        crop_w, crop_h = img_width - left, img_height - top
        pos = "north-west"
    else:
        # np.all(np.array([top_in, left_in, bottom_in, right_in])):
        return left, top, width, height, None

    return crop_left, crop_top, crop_w, crop_h, pos
