"""Spatial indexing and morton code utilities for precomputed annotations.

Handles chunk assignment of annotations to spatial grid cells,
compressed morton code encoding for sharded spatial index keys,
and multi-level spatial hierarchy construction with probabilistic
coarse-to-fine subsampling (per the neuroglancer spec).
"""

import math
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import product
from typing import Literal, NamedTuple, Optional

import numpy as np

from ._encoding import encode_multiple_annotations

# ── Data Structures ─────────────────────────────────────────────────


class SpatialLevel(NamedTuple):
    """One level in a multi-scale spatial index hierarchy.

    Attributes
    ----------
    key : str
        URL path prefix for this level (e.g. ``"spatial0"``).
    grid_shape : np.ndarray
        ``(rank,)`` int array — number of grid cells per dimension.
    chunk_size : np.ndarray
        ``(rank,)`` float array — physical size of each cell per dimension.
    limit : int
        Maximum annotations per chunk at this level (target for
        probabilistic subsampling).
    """

    key: str
    grid_shape: np.ndarray
    chunk_size: np.ndarray
    limit: int


@dataclass
class MultiscaleAssignment:
    """Result of assigning annotations to a multi-level spatial hierarchy.

    Attributes
    ----------
    levels : list[SpatialLevel]
        The spatial levels (coarsest first).
    row_assignments : list[list[tuple[int, tuple[int, ...]]]]
        Forward mapping. ``row_assignments[i]`` is a list of
        ``(level_index, cell_tuple)`` pairs where annotation *i* was
        emitted. For point annotations each inner list has exactly one
        entry; multi-point types may appear in multiple cells.
    level_chunks : dict[tuple[int, tuple[int, ...]], np.ndarray]
        Inverse mapping. Key is ``(level_index, cell_tuple)``, value is
        an int array of row indices assigned to that chunk.
    """

    levels: list[SpatialLevel]
    row_assignments: list[list[tuple[int, tuple[int, ...]]]]
    level_chunks: dict[tuple[int, tuple[int, ...]], np.ndarray] = field(
        default_factory=dict
    )


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


# ── Multi-scale Spatial Hierarchy ───────────────────────────────────


def build_spatial_levels(
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    n_annotations: int,
    limit: int = 5000,
    levels: Optional[list[SpatialLevel]] = None,
) -> list[SpatialLevel]:
    """Build spatial index level hierarchy from bounds and annotation count.

    If *levels* is provided it is returned directly (passthrough for
    advanced users). Otherwise a hierarchy is auto-computed by starting
    from a finest grid that targets *limit* annotations per cell, then
    coarsening by 2× until the grid shape is ``[1, 1, 1]``.

    Parameters
    ----------
    lower_bound : np.ndarray
        ``(rank,)`` lower bound.
    upper_bound : np.ndarray
        ``(rank,)`` upper bound.
    n_annotations : int
        Total annotation count (used to size the finest grid).
    limit : int
        Target max annotations per chunk at each level.
    levels : list[SpatialLevel], optional
        Explicit level configs. If given, returned as-is.

    Returns
    -------
    list[SpatialLevel]
        Levels ordered coarsest-first (``spatial0`` = coarsest).
    """
    if levels is not None:
        return levels

    extent = (upper_bound - lower_bound).astype(np.float64)
    extent = np.maximum(extent, 1.0)
    rank = len(extent)

    # Finest-level chunk size: target ~limit annotations per cell
    finest_chunk = auto_chunk_size(lower_bound, upper_bound, n_annotations, limit)
    finest_grid = np.maximum(np.ceil(extent / finest_chunk).astype(int), 1)

    # Build levels by coarsening from finest → [1,1,1]
    grids: list[np.ndarray] = [finest_grid]
    while np.any(grids[-1] > 1):
        coarser = np.maximum(np.ceil(grids[-1] / 2).astype(int), 1)
        grids.append(coarser)

    # Reverse so coarsest is first (spatial0 = [1,1,1])
    grids.reverse()

    result: list[SpatialLevel] = []
    for i, grid in enumerate(grids):
        cs = extent / grid
        result.append(
            SpatialLevel(
                key=f"spatial{i}",
                grid_shape=grid,
                chunk_size=cs,
                limit=limit,
            )
        )
    return result


def compute_multiscale_assignments(
    coordinates: np.ndarray,
    lower_bound: np.ndarray,
    levels: list[SpatialLevel],
    rng: Optional[np.random.Generator] = None,
) -> MultiscaleAssignment:
    """Assign annotations to a multi-level spatial hierarchy.

    Implements the neuroglancer probabilistic cascade: at each level
    (coarse → fine), remaining annotations are assigned to grid cells,
    then a random subset (capped by the level's ``limit``) is emitted.
    Un-emitted annotations cascade to the next finer level. The finest
    level always emits all remaining.

    Parameters
    ----------
    coordinates : np.ndarray
        ``(N, geometry_size)`` float32. For points ``(N, rank)``;
        for two-point types ``(N, 2*rank)``.
    lower_bound : np.ndarray
        ``(rank,)`` lower bound of the spatial grid.
    levels : list[SpatialLevel]
        Spatial levels, coarsest first.
    rng : np.random.Generator, optional
        Random generator for reproducibility. Falls back to default.

    Returns
    -------
    MultiscaleAssignment
        Contains the forward mapping ``row_assignments`` and inverse
        mapping ``level_chunks``.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(coordinates)
    remaining = np.ones(n, dtype=bool)

    row_assignments: list[list[tuple[int, tuple[int, ...]]]] = [[] for _ in range(n)]
    level_chunks: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)

    for level_idx, level in enumerate(levels):
        is_last = level_idx == len(levels) - 1
        remaining_indices = np.where(remaining)[0]
        if len(remaining_indices) == 0:
            break

        # Assign remaining annotations to cells at this level
        cell_to_rows = compute_chunk_assignments(
            coordinates[remaining_indices],
            lower_bound,
            level.chunk_size,
            level.grid_shape,
        )
        # Remap local→global indices
        cell_to_rows = {
            cell: remaining_indices[local_idx]
            for cell, local_idx in cell_to_rows.items()
        }

        if is_last:
            # Finest level: emit everything remaining
            for cell, indices in cell_to_rows.items():
                for idx in indices:
                    row_assignments[idx].append((level_idx, cell))
                    level_chunks[(level_idx, cell)].append(idx)
                    remaining[idx] = False
        else:
            # Compute maxCount across all cells at this level
            max_count = max(
                (len(indices) for indices in cell_to_rows.values()), default=0
            )
            if max_count == 0:
                continue

            prob = min(1.0, level.limit / max_count)

            for cell, indices in cell_to_rows.items():
                # Sample each annotation independently with probability prob
                mask = rng.random(len(indices)) < prob
                emitted = indices[mask]

                for idx in emitted:
                    row_assignments[idx].append((level_idx, cell))
                    level_chunks[(level_idx, cell)].append(idx)
                    remaining[idx] = False

    # Convert inverse lists to arrays
    level_chunks_arr = {
        key: np.array(val, dtype=np.intp) for key, val in level_chunks.items()
    }

    return MultiscaleAssignment(
        levels=levels,
        row_assignments=row_assignments,
        level_chunks=level_chunks_arr,
    )


def encode_multiscale_spatial_chunks(
    fixed_blocks: np.ndarray,
    ids: np.ndarray,
    assignment: MultiscaleAssignment,
) -> dict[tuple[int, tuple[int, ...]], bytes]:
    """Encode annotations into binary chunks across multiple spatial levels.

    Parameters
    ----------
    fixed_blocks : np.ndarray
        ``(N,)`` structured array of all annotations.
    ids : np.ndarray
        ``(N,)`` uint64 annotation IDs.
    assignment : MultiscaleAssignment
        The multi-scale assignment result.

    Returns
    -------
    dict[tuple[int, tuple[int, ...]], bytes]
        Maps ``(level_index, cell_tuple)`` to encoded binary chunk data.
    """
    chunks: dict[tuple[int, tuple[int, ...]], bytes] = {}
    for (level_idx, cell), indices in assignment.level_chunks.items():
        chunk_blocks = fixed_blocks[indices]
        chunk_ids = ids[indices]
        perm = np.random.permutation(len(chunk_blocks))
        chunks[(level_idx, cell)] = encode_multiple_annotations(
            chunk_blocks[perm], chunk_ids[perm]
        )
    return chunks


# ── Class-based Spatial Hierarchy API ───────────────────────────────


class _SpatialHierarchy(ABC):
    """Base class for spatial index hierarchy computation.

    Implements an sklearn-style fit/assign lifecycle:

    - ``fit(coordinates)`` computes bounds and delegates to
      ``_build_levels()`` to construct the grid hierarchy.
    - ``assign(coordinates)`` runs the shared coarse-to-fine
      probabilistic cascade using fitted parameters.
    - ``fit_assign(coordinates)`` is a convenience combining both.

    Subclasses override ``_build_levels()`` to define how the
    multi-level grid hierarchy is constructed.

    Parameters
    ----------
    rank : int
        Number of spatial dimensions (e.g. 3 for 3-D data).
    limit : int
        Target maximum annotations per chunk at each level.
    lower_bound : np.ndarray, optional
        Explicit lower bound. If ``None``, computed from data in ``fit()``.
    upper_bound : np.ndarray, optional
        Explicit upper bound. If ``None``, computed from data in ``fit()``.
    bound_padding : float
        Fractional padding applied to auto-computed bounds (not
        user-specified). E.g. ``0.05`` pads by 5% of extent.
    out_of_bounds : ``"error"`` | ``"warn"`` | ``"ignore"``
        How ``assign()`` handles coordinates outside fitted bounds.

    Attributes (after fit)
    ----------------------
    levels_ : list[SpatialLevel]
        The spatial levels (coarsest first).
    lower_bound_ : np.ndarray
        Resolved lower bound.
    upper_bound_ : np.ndarray
        Resolved upper bound.
    """

    def __init__(
        self,
        rank: int,
        limit: int = 5000,
        lower_bound: Optional[np.ndarray] = None,
        upper_bound: Optional[np.ndarray] = None,
        bound_padding: float = 0.0,
        out_of_bounds: Literal["error", "warn", "ignore"] = "warn",
    ) -> None:
        self.rank = rank
        self.limit = limit
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.bound_padding = bound_padding
        self.out_of_bounds = out_of_bounds

        # Fitted attributes (set by fit())
        self.levels_: Optional[list[SpatialLevel]] = None
        self.lower_bound_: Optional[np.ndarray] = None
        self.upper_bound_: Optional[np.ndarray] = None

    def _check_is_fitted(self) -> None:
        if self.levels_ is None:
            raise RuntimeError(
                f"{type(self).__name__} has not been fitted. Call fit() first."
            )

    def _check_coords(self, coordinates: np.ndarray) -> None:
        """Validate that coordinate dimensions are consistent with rank."""
        geom_size = coordinates.shape[1]
        if geom_size not in (self.rank, 2 * self.rank):
            raise ValueError(
                f"Coordinate columns ({geom_size}) inconsistent with rank "
                f"({self.rank}). Expected {self.rank} (point) or "
                f"{2 * self.rank} (two-point)."
            )

    def _compute_bounds(self, coordinates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute or adopt bounds from coordinates.

        Uses user-specified bounds if provided, otherwise computes from
        data. Applies ``bound_padding`` only to auto-computed bounds.
        Applies float32 precision adjustment to upper bound.
        """
        rank = self.rank
        geom_size = coordinates.shape[1]

        if geom_size == rank:
            data_lower = coordinates.min(axis=0)
            data_upper = coordinates.max(axis=0)
        else:
            coords_a = coordinates[:, :rank]
            coords_b = coordinates[:, rank:]
            data_lower = np.minimum(coords_a, coords_b).min(axis=0)
            data_upper = np.maximum(coords_a, coords_b).max(axis=0)

        if self.lower_bound is not None:
            lower = np.asarray(self.lower_bound, dtype=np.float64).copy()
        else:
            lower = data_lower.astype(np.float64)

        if self.upper_bound is not None:
            upper = np.asarray(self.upper_bound, dtype=np.float64).copy()
        else:
            upper = data_upper.astype(np.float64)

        # Apply padding only to auto-computed bounds
        if self.bound_padding > 0:
            extent = upper - lower
            extent = np.maximum(extent, 1.0)
            pad = extent * self.bound_padding
            if self.lower_bound is None:
                lower -= pad
            if self.upper_bound is None:
                upper += pad

        # Ensure non-zero extent
        extent = upper - lower
        extent = np.maximum(extent, 1.0)
        upper = lower + extent

        # Float32 precision adjustment
        upper = np.nextafter(upper, np.inf)

        return lower, upper

    def _check_oob(self, coordinates: np.ndarray) -> None:
        """Check for out-of-bounds coordinates and handle per policy."""
        rank = self.rank
        geom_size = coordinates.shape[1]

        # Extract all coordinate columns for OOB check
        if geom_size == rank:
            all_coords = coordinates
        else:
            all_coords = np.concatenate(
                [coordinates[:, :rank], coordinates[:, rank:]], axis=0
            )

        oob = (all_coords < self.lower_bound_) | (all_coords >= self.upper_bound_)
        if oob.any():
            if geom_size == rank:
                n_oob = oob.any(axis=1).sum()
            else:
                # For multi-point: check per-annotation (either point OOB)
                n_half = len(coordinates)
                oob_a = oob[:n_half].any(axis=1)
                oob_b = oob[n_half:].any(axis=1)
                n_oob = (oob_a | oob_b).sum()

            msg = f"{n_oob} annotation(s) have coordinates outside fitted bounds"
            if self.out_of_bounds == "error":
                raise ValueError(msg)
            elif self.out_of_bounds == "warn":
                warnings.warn(msg, stacklevel=3)

    def fit(self, coordinates: np.ndarray) -> "_SpatialHierarchy":
        """Fit the spatial hierarchy to coordinate data.

        Computes bounds (or adopts user-specified ones), then delegates
        to ``_build_levels()`` to construct the grid hierarchy.

        Parameters
        ----------
        coordinates : np.ndarray
            ``(N, geometry_size)`` float array.

        Returns
        -------
        self
        """
        self._check_coords(coordinates)
        self.lower_bound_, self.upper_bound_ = self._compute_bounds(coordinates)
        self.levels_ = self._build_levels(coordinates)
        return self

    @abstractmethod
    def _build_levels(self, coordinates: np.ndarray) -> list[SpatialLevel]:
        """Build the spatial level hierarchy.

        Called during ``fit()`` after bounds are resolved. Access
        ``self.lower_bound_``, ``self.upper_bound_``, ``self.limit``,
        and the raw coordinates to construct levels.

        Parameters
        ----------
        coordinates : np.ndarray
            ``(N, geometry_size)`` float array.

        Returns
        -------
        list[SpatialLevel]
            Levels ordered coarsest-first.
        """
        ...

    def assign(
        self,
        coordinates: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> MultiscaleAssignment:
        """Assign annotations to the fitted spatial hierarchy.

        Runs the coarse-to-fine probabilistic cascade using the
        levels computed during ``fit()``.

        Parameters
        ----------
        coordinates : np.ndarray
            ``(N, geometry_size)`` float array.
        rng : np.random.Generator, optional
            Random generator for reproducibility.

        Returns
        -------
        MultiscaleAssignment
        """
        self._check_is_fitted()
        self._check_coords(coordinates)
        self._check_oob(coordinates)
        return compute_multiscale_assignments(
            coordinates, self.lower_bound_, self.levels_, rng=rng
        )

    def fit_assign(
        self,
        coordinates: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> MultiscaleAssignment:
        """Fit the hierarchy and assign in one step.

        Parameters
        ----------
        coordinates : np.ndarray
            ``(N, geometry_size)`` float array.
        rng : np.random.Generator, optional
            Random generator for reproducibility.

        Returns
        -------
        MultiscaleAssignment
        """
        return self.fit(coordinates).assign(coordinates, rng=rng)


class _UniformHierarchy(_SpatialHierarchy):
    """Spatial hierarchy with uniform subdivision across all dimensions.

    Builds the finest grid targeting ``limit`` annotations per cell
    (assuming uniform distribution), then coarsens by halving all
    dimensions until ``[1, 1, ..., 1]``.

    Parameters
    ----------
    chunk_size : float or array-like, optional
        Explicit finest-level chunk size. If provided, overrides the
        auto-computed chunk size.
    **kwargs
        Passed to :class:`SpatialHierarchy`.
    """

    def __init__(
        self,
        chunk_size: Optional[np.ndarray | float] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.chunk_size = chunk_size

    def _build_levels(self, coordinates: np.ndarray) -> list[SpatialLevel]:
        extent = (self.upper_bound_ - self.lower_bound_).astype(np.float64)
        extent = np.maximum(extent, 1.0)
        n_annotations = len(coordinates)

        if self.chunk_size is not None:
            rank = self.rank
            if isinstance(self.chunk_size, (int, float)):
                finest_cs = np.full(rank, float(self.chunk_size))
            else:
                finest_cs = np.asarray(self.chunk_size, dtype=np.float64)
            finest_grid = np.maximum(np.ceil(extent / finest_cs).astype(int), 1)
            # Snap upper bound to grid
            self.upper_bound_ = self.lower_bound_ + finest_grid * finest_cs
            extent = self.upper_bound_ - self.lower_bound_
        else:
            finest_chunk = auto_chunk_size(
                self.lower_bound_, self.upper_bound_, n_annotations, self.limit
            )
            finest_grid = np.maximum(np.ceil(extent / finest_chunk).astype(int), 1)

        # Build levels by coarsening from finest → [1,1,...,1]
        grids: list[np.ndarray] = [finest_grid]
        while np.any(grids[-1] > 1):
            coarser = np.maximum(np.ceil(grids[-1] / 2).astype(int), 1)
            grids.append(coarser)

        # Reverse so coarsest is first (spatial0 = [1,1,...,1])
        grids.reverse()

        result: list[SpatialLevel] = []
        for i, grid in enumerate(grids):
            cs = extent / grid
            result.append(
                SpatialLevel(
                    key=f"spatial{i}",
                    grid_shape=grid,
                    chunk_size=cs,
                    limit=self.limit,
                )
            )
        return result


class _IsotropicHierarchy(_SpatialHierarchy):
    """Spatial hierarchy with isotropy-aware subdivision.

    Builds levels coarse-to-fine per the neuroglancer spec: at each
    refinement step, each dimension is halved only if doing so produces
    a more spatially isotropic chunk. Terminates when the expected
    annotation density per cell is at or below ``limit``.

    Parameters
    ----------
    **kwargs
        Passed to :class:`SpatialHierarchy`.
    """

    def _build_levels(self, coordinates: np.ndarray) -> list[SpatialLevel]:
        extent = (self.upper_bound_ - self.lower_bound_).astype(np.float64)
        extent = np.maximum(extent, 1.0)
        rank = self.rank
        n_annotations = len(coordinates)

        # Start with coarsest level: [1, 1, ..., 1]
        grid = np.ones(rank, dtype=int)
        grids: list[np.ndarray] = [grid.copy()]

        # Refine until expected density per cell <= limit
        while True:
            n_cells = int(np.prod(grid))
            expected_per_cell = n_annotations / n_cells
            if expected_per_cell <= self.limit:
                break

            # Per the spec: for each dimension, the finer chunk_size
            # should be equal to or half of the coarser. Halve whichever
            # results in a more isotropic chunk. Strategy: halve all
            # dimensions whose chunk_size equals the current maximum
            # (within tolerance), producing more cubic chunks.
            chunk_size = extent / grid
            cs_max = chunk_size.max()

            # Halve dimensions at or near the maximum chunk_size
            # (within 1% tolerance to handle floating point)
            candidate = grid.copy()
            dims_to_halve = chunk_size >= cs_max * 0.99
            if not dims_to_halve.any():
                # Fallback: halve the single largest
                dims_to_halve = np.zeros(rank, dtype=bool)
                dims_to_halve[np.argmax(chunk_size)] = True

            candidate[dims_to_halve] = grid[dims_to_halve] * 2

            grid = candidate
            grids.append(grid.copy())

        # Snap upper bound to finest grid
        finest_grid = grids[-1]
        self.upper_bound_ = self.lower_bound_ + finest_grid * (extent / finest_grid)

        result: list[SpatialLevel] = []
        for i, g in enumerate(grids):
            cs = extent / g
            result.append(
                SpatialLevel(
                    key=f"spatial{i}",
                    grid_shape=g,
                    chunk_size=cs,
                    limit=self.limit,
                )
            )
        return result
