"""Binary encoding for neuroglancer precomputed annotations.

Handles geometry + property encoding into numpy structured arrays,
single-annotation encoding (for by_id index), and multiple-annotation
encoding (for spatial and relationship indexes).

Reference: neuroglancer/src/datasource/precomputed/annotations.md
"""

import struct
from collections.abc import Sequence
from typing import Literal, Union

import numpy as np
from neuroglancer import viewer_state

AnnotationType = Literal["point", "line", "axis_aligned_bounding_box", "ellipsoid"]

# Maps property type string to (numpy dtype args, alignment).
# Alignment determines sort order for padding minimization per the spec:
#   4-byte types first, then 2-byte, then 1-byte, then pad to 4-byte boundary.
_PROPERTY_DTYPES: dict[
    str, tuple[Union[tuple[str], tuple[str, tuple[int, ...]]], int]
] = {
    "uint8": (("|u1",), 1),
    "uint16": (("<u2",), 2),
    "uint32": (("<u4",), 4),
    "int8": (("|i1",), 1),
    "int16": (("<i2",), 2),
    "int32": (("<i4",), 4),
    "float32": (("<f4",), 4),
    "rgb": (("|u1", (3,)), 1),
    "rgba": (("|u1", (4,)), 1),
}


def _get_dtype_for_geometry(annotation_type: AnnotationType, rank: int):
    """Build the numpy dtype entries for the geometry portion of an annotation."""
    geometry_size = rank if annotation_type == "point" else 2 * rank
    return [("geometry", "<f4", geometry_size)]


def _get_dtype_for_properties(
    properties: Sequence[viewer_state.AnnotationPropertySpec],
) -> list:
    """Build the numpy dtype entries for annotation properties.

    Properties are ordered by alignment (4-byte, 2-byte, 1-byte) per the spec,
    with padding to 4-byte boundary at the end.
    """
    dtype = []
    offset = 0
    for p in properties:
        dtype_entry, alignment = _PROPERTY_DTYPES[p.type]
        dtype.append((p.id, *dtype_entry))
        size = np.dtype(dtype[-1:]).itemsize
        offset += size
    alignment = 4
    if offset % alignment:
        padded_offset = (offset + alignment - 1) // alignment * alignment
        padding = padded_offset - offset
        dtype.append((f"_padding{offset}", "|u1", (padding,)))
    return dtype


def sort_properties(
    properties: Sequence[viewer_state.AnnotationPropertySpec],
) -> list[viewer_state.AnnotationPropertySpec]:
    """Sort properties by descending alignment (4-byte first, then 2, then 1).

    This matches the binary encoding order required by the spec.
    """
    return sorted(properties, key=lambda p: -_PROPERTY_DTYPES[p.type][1])


def build_dtype(
    annotation_type: AnnotationType,
    rank: int,
    properties: Sequence[viewer_state.AnnotationPropertySpec],
) -> np.dtype:
    """Build the full numpy structured dtype for fixed-block encoding.

    The fixed block = geometry + properties + padding. This block is identical
    across by_id, spatial, and relationship indexes.
    """
    entries = _get_dtype_for_geometry(
        annotation_type, rank
    ) + _get_dtype_for_properties(properties)
    return np.dtype(entries)


def encode_fixed_blocks(
    coordinates: np.ndarray,
    properties: dict[str, np.ndarray],
    dtype: np.dtype,
) -> np.ndarray:
    """Encode annotation geometry and properties into a structured array.

    Parameters
    ----------
    coordinates : np.ndarray
        (N, geometry_size) array of float32 coordinates.
        For points: (N, rank). For lines/bbox/ellipsoid: (N, 2*rank).
    properties : dict[str, np.ndarray]
        Maps property id to (N,) array of values.
    dtype : np.dtype
        Structured dtype from build_dtype().

    Returns
    -------
    np.ndarray
        (N,) structured array. Each element's .tobytes() is the fixed-block
        encoding for one annotation.
    """
    n = len(coordinates)
    arr = np.zeros(n, dtype=dtype)
    arr["geometry"] = coordinates
    for name, values in properties.items():
        arr[name] = values
    return arr


def encode_by_id_entry(
    fixed_block_bytes: bytes,
    relationships: list[list[int]],
) -> bytes:
    """Encode a single annotation for the by_id index.

    Appends variable-length relationship data after the fixed block.

    Parameters
    ----------
    fixed_block_bytes : bytes
        The fixed-block bytes for this annotation (from structured array .tobytes()).
    relationships : list[list[int]]
        For each relationship defined in the collection, a list of related segment IDs.

    Returns
    -------
    bytes
        Complete binary encoding for one by_id entry.
    """
    parts = [fixed_block_bytes]
    for rel_ids in relationships:
        parts.append(struct.pack("<I", len(rel_ids)))
        for rid in rel_ids:
            parts.append(struct.pack("<Q", rid))
    return b"".join(parts)


def encode_by_id_entries(
    fixed_blocks: np.ndarray,
    relationships: list[list[list[int]]] | None,
) -> list[bytes]:
    """Encode all annotations for the by_id index.

    Parameters
    ----------
    fixed_blocks : np.ndarray
        (N,) structured array from encode_fixed_blocks().
    relationships : list[list[list[int]]] or None
        relationships[r][i] = list of segment IDs for relationship r, annotation i.
        None if no relationships.

    Returns
    -------
    list[bytes]
        One bytes object per annotation.
    """
    n = len(fixed_blocks)
    itemsize = fixed_blocks.dtype.itemsize
    raw = fixed_blocks.tobytes()

    if relationships is None or len(relationships) == 0:
        return [raw[i * itemsize : (i + 1) * itemsize] for i in range(n)]

    num_relationships = len(relationships)
    entries = []
    for i in range(n):
        row_rels = [relationships[r][i] for r in range(num_relationships)]
        entry = encode_by_id_entry(
            raw[i * itemsize : (i + 1) * itemsize],
            row_rels,
        )
        entries.append(entry)
    return entries


def encode_multiple_annotations(
    fixed_blocks: np.ndarray,
    ids: np.ndarray,
) -> bytes:
    """Encode a list of annotations in the multiple-annotation format.

    Used for spatial index chunks and relationship index entries.
    Format: uint64(count) + [fixed_block × N] + [uint64(id) × N]

    Parameters
    ----------
    fixed_blocks : np.ndarray
        (N,) structured array of fixed blocks for the annotations in this chunk.
    ids : np.ndarray
        (N,) uint64 array of annotation IDs.

    Returns
    -------
    bytes
        Complete binary encoding for a multi-annotation chunk.
    """
    n = len(fixed_blocks)
    parts = [
        struct.pack("<Q", n),
        fixed_blocks.tobytes(),
        np.ascontiguousarray(ids, dtype="<u8").tobytes(),
    ]
    return b"".join(parts)
