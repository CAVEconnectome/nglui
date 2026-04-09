"""Precomputed annotation writing for neuroglancer.

Write pandas DataFrames or annotation arrays to the neuroglancer
precomputed annotation format, with support for local and cloud storage.

Public API
----------
AnnotationDataFrameWriter
    High-level: map DataFrame columns to precomputed annotations.
PrecomputedAnnotationWriter
    Low-level: bulk array and per-row annotation writing.
"""

from .dataframe_writer import AnnotationDataFrameWriter
from .writer import PrecomputedAnnotationWriter
from .spatial_index_tree import SpatialIndexTree

__all__ = [
    "AnnotationDataFrameWriter",
    "PrecomputedAnnotationWriter",
    "SpatialIndexTree",
]
