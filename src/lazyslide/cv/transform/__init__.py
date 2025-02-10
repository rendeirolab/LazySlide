"""This module is highly inspired by both torchvison and pathml"""

from .compose import TissueDetectionHE

from .mods import (
    MedianBlur,
    GaussianBlur,
    BoxBlur,
    MorphOpen,
    MorphClose,
    BinaryThreshold,
    ArtifactFilterThreshold,
    Compose,
)
