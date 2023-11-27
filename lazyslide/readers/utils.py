import numpy as np
from numba import njit


@njit
def get_crop_xy_wh(img_width, img_height, x, y, width, height):
    x1_in = 0 <= x <= img_width
    y1_in = 0 <= y <= img_height
    x2_in = 0 <= (x + width) <= img_width
    y2_in = 0 <= (y + height) <= img_height
    x1_out, x2_out, y1_out, y2_out = not x1_in, not x2_in, not y1_in, not y2_in

    # If extract from region outside image
    if (x1_out and x2_out) or (y1_out and y2_out):
        raise RuntimeError("Extracting region that are completely outside image.")

    if np.all(np.array([x1_in, y1_in, x2_in, y2_in])):
        return x, y, width, height, None
    elif x1_out and x2_in and y1_out and y2_in:
        crop_x, crop_y = 0, 0
        crop_w, crop_h = width + x, height + y
        pos = "south-east"
    elif x1_out and x2_in and y1_in and y2_in:
        crop_x, crop_y = 0, y
        crop_w, crop_h = width + x, height
        pos = "east"
    elif x1_out and x2_in and y1_in and y2_out:
        crop_x, crop_y = 0, y
        crop_w, crop_h = width + x, img_height - y
        pos = "north-east"
    elif x1_in and x2_in and y1_out and y2_in:
        crop_x, crop_y = x, 0
        crop_w, crop_h = width, height + y
        pos = "south"
    elif x1_in and x2_in and y1_in and y2_out:
        crop_x, crop_y = x, y
        crop_w, crop_h = width, img_height - y
        pos = "north"
    elif x1_in and x2_out and y1_out and y2_in:
        crop_x, crop_y = x, 0
        crop_w, crop_h = img_width - x, height + y
        pos = "south-west"
    elif x1_in and x2_out and y1_in and y2_in:
        crop_x, crop_y = x, y
        crop_w, crop_h = img_width - x, height
        pos = "west"
    else:
        crop_x, crop_y = x, y
        crop_w, crop_h = img_width - x, img_height - y
        pos = "north-west"

    return crop_x, crop_y, crop_w, crop_h, pos
