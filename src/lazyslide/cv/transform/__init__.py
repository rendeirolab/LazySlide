"""This module is highly inspired by both torchvison and pathml"""

from .compose import TissueDetectionHE
from .mods import (
    ArtifactFilterThreshold,
    BinaryThreshold,
    BoxBlur,
    Compose,
    GaussianBlur,
    MedianBlur,
    MorphClose,
    MorphOpen,
)

__all__ = [
    "Compose",
    "BinaryThreshold",
    "BoxBlur",
    "GaussianBlur",
    "MedianBlur",
    "MorphClose",
    "MorphOpen",
    "ArtifactFilterThreshold",
    "TissueDetectionHE",
]
