"""Shapenet Part dataset."""

from .part import (
    LABEL_SPLITS,
    NUM_OBJECT_CLASSES,
    NUM_PART_CLASSES,
    POINT_CLASS_FREQ,
    ShapenetPart2017,
    ShapenetPart2017Config,
    part_class_indices,
)

__all__ = [
    "LABEL_SPLITS",
    "NUM_OBJECT_CLASSES",
    "NUM_PART_CLASSES",
    "POINT_CLASS_FREQ",
    "ShapenetPart2017",
    "ShapenetPart2017Config",
    "part_class_indices",
]
