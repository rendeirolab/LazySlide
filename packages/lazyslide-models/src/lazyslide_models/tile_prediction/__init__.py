from .cv_features import (
    Brightness,
    Canny,
    Contrast,
    Entropy,
    HaralickTexture,
    Saturation,
    Sharpness,
    Sobel,
    SplitRGB,
)
from .focuslitenn import FocusLiteNN
from .pathprofiler_qc import PathProfilerQC
from .spider import (
    Spider,
    SpiderBreast,
    SpiderColorectal,
    SpiderSkin,
    SpiderThorax,
)

CV_FEATURES = {
    "split_rgb": SplitRGB,
    "brightness": Brightness,
    "contrast": Contrast,
    "sobel": Sobel,
    "canny": Canny,
    "sharpness": Sharpness,
    "entropy": Entropy,
    "saturation": Saturation,
    "haralick_texture": HaralickTexture,
}


__all__ = [
    "CV_FEATURES",
    "FocusLiteNN",
    "PathProfilerQC",
    "Spider",
    "SpiderSkin",
    "SpiderBreast",
    "SpiderColorectal",
    "SpiderThorax",
    "Brightness",
    "Contrast",
    "Sharpness",
    "Sobel",
    "Canny",
    "Entropy",
    "Saturation",
    "HaralickTexture",
    "SplitRGB",
]
