# This is copied from https://github.com/scikit-image/scikit-image/blob/v0.24.0/skimage/util/dtype.py
import warnings

import numpy as np

_integer_types = (
    np.int8,
    np.byte,
    np.int16,
    np.short,
    np.int32,
    np.int64,
    np.longlong,
    np.int_,
    np.intp,
    np.intc,
    int,
    np.uint8,
    np.ubyte,
    np.uint16,
    np.ushort,
    np.uint32,
    np.uint64,
    np.ulonglong,
    np.uint,
    np.uintp,
    np.uintc,
)
_integer_ranges = {t: (np.iinfo(t).min, np.iinfo(t).max) for t in _integer_types}
dtype_range = {
    bool: (False, True),
    np.bool_: (False, True),
    float: (-1, 1),
    np.float16: (-1, 1),
    np.float32: (-1, 1),
    np.float64: (-1, 1),
}

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # np.bool8 is a deprecated alias of np.bool_
    if hasattr(np, "bool8"):
        dtype_range[np.bool8] = (False, True)

dtype_range.update(_integer_ranges)

_supported_types = list(dtype_range.keys())


def dtype_limits(image, clip_negative=False):
    """Return intensity limits, i.e. (min, max) tuple, of the image's dtype.

    Parameters
    ----------
    image : ndarray
        Input image.
    clip_negative : bool, optional
        If True, clip the negative range (i.e. return 0 for min intensity)
        even if the image dtype allows negative values.

    Returns
    -------
    imin, imax : tuple
        Lower and upper intensity limits.
    """
    imin, imax = dtype_range[image.dtype.type]
    if clip_negative:
        imin = 0
    return imin, imax
