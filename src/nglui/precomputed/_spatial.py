"""Spatial indexing and morton code utilities for precomputed annotations.

Handles chunk assignment of annotations to spatial grid cells and
compressed morton code encoding for sharded spatial index keys.
"""

import math
from itertools import product

import numpy as np

from ._encoding import encode_multiple_annotations


def compute_chunk_assignments(
    coordinates: np.ndarray,
    lower_bound: np.ndarray,
    chunk_size: np.ndarray,
    num_chunks: np.ndarray,
) -> dict[tuple[int, ...], np.ndarray]:
    """Assign annotations to spatial grid cells based on their coordinates.

    For point annotations, each point falls in exactly one cell.
    For multi-point annotations (line, bbox, ellipsoid), an annotation
    may intersect multiple cells.

    Parameters
    ----------
    coordinates : np.ndarray
        (N, geometry_size) float32 array. For points: (N, rank).
        For line/bbox/ellipsoid: (N, 2*rank).
    lower_bound : np.ndarray
        (rank,) lower bound of the spatial grid.
    chunk_size : np.ndarray
        (rank,) size of each grid cell.
    num_chunks : np.ndarray
        (rank,) number of cells along each dimension.

    Returns
    -------
    dict[tuple[int, ...], np.ndarray]
        Maps cell coordinate tuple to array of annotation indices in that cell.
    """
    rank = len(lower_bound)
    n = len(coordinates)
    geom_size = coordinates.shape[1]

    if geom_size == rank:
        # Point annotations: single position
        cell_indices = ((coordinates - lower_bound) / chunk_size).astype(np.int32)
        cell_indices = np.clip(cell_indices, 0, num_chunks - 1)

        assignments: dict[tuple[int, ...], list[int]] = {}
        for i in range(n):
            key = tuple(cell_indices[i])
            if key not in assignments:
                assignments[key] = []
            assignments[key].append(i)
    else:
        # Two-point annotations: min/max bounding box across both points
        coords_a = coordinates[:, :rank]
        coords_b = coordinates[:, rank:]
        lo = np.minimum(coords_a, coords_b)
        hi = np.maximum(coords_a, coords_b)

        cell_lo = ((lo - lower_bound) / chunk_size).astype(np.int32)
        cell_hi = ((hi - lower_bound) / chunk_size).astype(np.int32)
        cell_lo = np.clip(cell_lo, 0, num_chunks - 1)
        cell_hi = np.clip(cell_hi, 0, num_chunks - 1)

        assignments = {}
        for i in range(n):
            ranges = [range(cell_lo[i, d], cell_hi[i, d] + 1) for d in range(rank)]
            for cell in product(*ranges):
                if cell not in assignments:
                    assignments[cell] = []
                assignments[cell].append(i)

    return {k: np.array(v, dtype=np.intp) for k, v in assignments.items()}


def encode_spatial_chunks(
    fixed_blocks: np.ndarray,
    ids: np.ndarray,
    chunk_assignments: dict[tuple[int, ...], np.ndarray],
) -> dict[tuple[int, ...], bytes]:
    """Encode annotations into binary spatial index chunks.

    Parameters
    ----------
    fixed_blocks : np.ndarray
        (N,) structured array of all annotations.
    ids : np.ndarray
        (N,) uint64 annotation IDs.
    chunk_assignments : dict[tuple[int, ...], np.ndarray]
        Maps cell coords to arrays of annotation indices.

    Returns
    -------
    dict[tuple[int, ...], bytes]
        Maps cell coords to encoded binary chunk data.
    """
    chunks = {}
    for cell, indices in chunk_assignments.items():
        chunk_blocks = fixed_blocks[indices]
        chunk_ids = ids[indices]
        # Spec says spatial index should be ordered randomly
        perm = np.random.permutation(len(chunk_blocks))
        chunks[cell] = encode_multiple_annotations(chunk_blocks[perm], chunk_ids[perm])
    return chunks


def compressed_morton_code(gridpt, grid_size):
    """Convert grid point(s) to compressed morton code (Z-order curve).

    Parameters
    ----------
    gridpt : array-like
        (3,) single grid point or (N, 3) array of grid points.
    grid_size : array-like
        (3,) number of cells along each dimension.

    Returns
    -------
    np.uint64 or np.ndarray of uint64
        Morton code(s).
    """
    if hasattr(gridpt, "__len__") and len(gridpt) == 0:
        return np.zeros((0,), dtype=np.uint64)

    gridpt = np.asarray(gridpt, dtype=np.uint32)
    single_input = False
    if gridpt.ndim == 1:
        gridpt = np.atleast_2d(gridpt)
        single_input = True

    code = np.zeros((gridpt.shape[0],), dtype=np.uint64)
    num_bits = [math.ceil(math.log2(max(size, 2))) for size in grid_size]
    j = np.uint64(0)
    one = np.uint64(1)

    for i in range(max(num_bits)):
        for dim in range(len(grid_size)):
            if 2**i < grid_size[dim]:
                bit = ((np.uint64(gridpt[:, dim]) >> np.uint64(i)) & one) << j
                code |= bit
                j += one

    if single_input:
        return code[0]
    return code


def auto_chunk_size(
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    n_annotations: int,
    target_per_chunk: int = 500,
) -> np.ndarray:
    """Compute a reasonable chunk size for the spatial index.

    Targets approximately `target_per_chunk` annotations per chunk,
    assuming uniform spatial distribution.

    Parameters
    ----------
    lower_bound : np.ndarray
        (rank,) lower bound.
    upper_bound : np.ndarray
        (rank,) upper bound.
    n_annotations : int
        Total number of annotations.
    target_per_chunk : int
        Target annotations per chunk.

    Returns
    -------
    np.ndarray
        (rank,) chunk size.
    """
    extent = upper_bound - lower_bound
    extent = np.maximum(extent, 1.0)  # avoid zero extents

    if n_annotations <= target_per_chunk:
        return extent.copy()

    # Number of chunks needed along each axis (uniform subdivision)
    n_chunks_total = max(1, n_annotations / target_per_chunk)
    rank = len(extent)
    n_per_dim = max(1, int(round(n_chunks_total ** (1.0 / rank))))

    chunk_size = extent / n_per_dim
    return chunk_size
