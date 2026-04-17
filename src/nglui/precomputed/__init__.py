"""Precomputed annotation writing for neuroglancer.

Write pandas DataFrames or annotation arrays to the neuroglancer
precomputed annotation format, with support for local and cloud storage.

Public API
----------
PointAnnotationWriter
    High-level: map DataFrame columns to precomputed point annotations.
LineAnnotationWriter
    High-level: map DataFrame columns to precomputed line annotations.
BoundingBoxAnnotationWriter
    High-level: map DataFrame columns to precomputed bounding box annotations.
EllipsoidAnnotationWriter
    High-level: map DataFrame columns to precomputed ellipsoid annotations.
PrecomputedAnnotationWriter
    Low-level: bulk array and per-row annotation writing.
"""

from .dataframe_writer import (
    BoundingBoxAnnotationWriter,
    EllipsoidAnnotationWriter,
    LineAnnotationWriter,
    PointAnnotationWriter,
)
from .writer import PrecomputedAnnotationWriter

__all__ = [
    "BoundingBoxAnnotationWriter",
    "EllipsoidAnnotationWriter",
    "LineAnnotationWriter",
    "PointAnnotationWriter",
    "PrecomputedAnnotationWriter",
]
