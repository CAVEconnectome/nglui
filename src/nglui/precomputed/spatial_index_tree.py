from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SpatialAssignment:
    """Records which level and cell a point was emitted at."""

    level: int
    cell: tuple[int, ...]


class SpatialIndexTree:
    """
    Builds and applies a neuroglancer-style spatial index hierarchy.

    Parameters
    ----------
    target_limit : int
        The target number of points per cell. The actual number in a tree from
        transforming data may be higher due to the randomised subsampling strategy.
    max_levels : int
        The maximum number of levels in the hierarchy.
    subsample : int, optional
        The number of points to subsample for building the hierarchy.
    seed : int
        The random seed for reproducibility.

    Attributes (available after fit)
    --------------------------------
    metadata_ : list[dict]
        Per-level dicts with keys "key", "grid_shape", "chunk_size", "limit",
        matching the neuroglancer info JSON "spatial" array format.
    lower_bound_ : np.ndarray
    upper_bound_ : np.ndarray
    grid_shapes_ : list[np.ndarray]
        The grid shape at each level.
    """

    def __init__(
        self,
        target_limit: int = 4096,
        max_levels: int = 20,
        subsample: Optional[int] = 10000,
        seed: int = 0,
    ):
        self.target_limit = target_limit
        self.max_levels = max_levels
        self.subsample = subsample
        self.seed = seed

        # Set after fit
        self.metadata_: list[dict] = []
        self.lower_bound_: Optional[np.ndarray] = None
        self.upper_bound_: Optional[np.ndarray] = None
        self.grid_shapes_: list[np.ndarray] = []
        self._chunk_sizes: list[np.ndarray] = []
        self._fitted = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_chunk_size(self, grid_shape: np.ndarray) -> np.ndarray:
        return (self.upper_bound_ - self.lower_bound_) / grid_shape.astype(np.float64)

    def _cell_of(self, pts: np.ndarray, level: int) -> np.ndarray:
        """Return (len(pts), rank) int array of cell coordinates for a level."""
        cs = self._chunk_sizes[level]
        gs = self.grid_shapes_[level]
        coords = ((pts - self.lower_bound_) / cs).astype(int)
        np.clip(coords, 0, gs - 1, out=coords)
        return coords

    @staticmethod
    def _group_by_cell(
        indices: np.ndarray,
        cell_coords: np.ndarray,
    ) -> dict[tuple, np.ndarray]:
        """Group point indices by their cell coordinate tuple."""
        if len(indices) == 0:
            return {}
        groups: dict[tuple, list[int]] = {}
        for idx, coord in zip(indices, cell_coords):
            key = tuple(coord)
            groups.setdefault(key, []).append(idx)
        return {k: np.array(v, dtype=int) for k, v in groups.items()}

    def _build_grid_shapes(self, rank: int) -> list[np.ndarray]:
        """
        Generate the sequence of grid shapes from coarse to fine.

        Level 0 is all-ones. Each subsequent level doubles the axis with
        the largest physical chunk size (most anisotropic), producing
        increasingly isotropic cells.
        """
        extent = self.upper_bound_ - self.lower_bound_
        shapes: list[np.ndarray] = [np.ones(rank, dtype=int)]
        for _ in range(self.max_levels - 1):
            prev = shapes[-1]
            cs = extent / prev.astype(np.float64)
            axis = int(np.argmax(cs))
            new = prev.copy()
            new[axis] *= 2
            shapes.append(new)
        return shapes

    def _ensure_fitted(self):
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        points: np.ndarray,
        lower_bound: Optional[np.ndarray] = None,
        upper_bound: Optional[np.ndarray] = None,
        n_samples: Optional[int] = None,
    ) -> tuple[list[dict], list[SpatialAssignment]]:
        """
        Compute the spatial hierarchy from data using the spec's
        randomised subsampling algorithm.

        Parameters
        ----------
        points : (N, rank) array
        lower_bound, upper_bound : (rank,) arrays, optional
            Defaults derived from data if not provided.
        n_samples : int, optional
            The expected number of samples in the overall hierarchy. Used to estimate
            the sampling probability at each level. Defaults to len(points). Use this
            if fitting on a subset of the data but want the hierarchy to reflect the
            full dataset with approximately `target_limit` points per cell.

        Returns
        -------
        metadata_ : list[dict]
            The "spatial" array for the info JSON.
        assignments : list[SpatialAssignment]
            Per-point level and cell where it was emitted.
        """
        points = np.asarray(points, dtype=np.float64)
        rank = points.shape[1]

        # here n is the expected number of points in the overall hierarchy
        if n_samples is None:
            n_expected = len(points)
        else:
            n_expected = n_samples

        n_subsamples = self.subsample if self.subsample is not None else n_expected
        n_subsamples = min(n_expected, n_subsamples)
        subsample_ratio = n_subsamples / n_expected
        limit = int(np.ceil(subsample_ratio * self.target_limit))

        rng = np.random.default_rng(self.seed)
        # this also handles a shuffle just once
        points = points[rng.choice(len(points), size=n_subsamples, replace=False)]

        self.lower_bound_ = (
            np.asarray(lower_bound, dtype=np.float64)
            if lower_bound is not None
            else points.min(axis=0)
        )
        self.upper_bound_ = (
            np.asarray(upper_bound, dtype=np.float64)
            if upper_bound is not None
            else points.max(axis=0) + 1e-6
        )

        self.grid_shapes_ = self._build_grid_shapes(rank)
        self._chunk_sizes = [self._compute_chunk_size(gs) for gs in self.grid_shapes_]

        assignments: list[Optional[SpatialAssignment]] = [None] * n_subsamples

        # Level-0 grouping
        all_idx = np.arange(n_subsamples, dtype=int)
        remaining = self._group_by_cell(all_idx, self._cell_of(points, 0))

        self.metadata_ = []

        for level in range(len(self.grid_shapes_)):
            if not remaining:
                break

            max_count = max(len(v) for v in remaining.values())
            if max_count == 0:
                break

            prob = min(1.0, limit / max_count)
            leftovers: list[np.ndarray] = []

            for cell, pt_idx in remaining.items():
                mask = rng.random(len(pt_idx)) < prob
                for i in pt_idx[mask]:
                    assignments[i] = SpatialAssignment(level=level, cell=cell)
                kept = pt_idx[~mask]
                if len(kept):
                    leftovers.append(kept)

            self.metadata_.append(
                {
                    "key": f"spatial{level}",
                    "grid_shape": self.grid_shapes_[level].tolist(),
                    "chunk_size": self._chunk_sizes[level].tolist(),
                    "limit": limit,
                }
            )

            # Re-bin leftovers into next level's finer grid
            if level + 1 < len(self.grid_shapes_) and leftovers:
                leftover = np.concatenate(leftovers)
                remaining = self._group_by_cell(
                    leftover, self._cell_of(points[leftover], level + 1)
                )
            else:
                remaining = {}

        # Any stragglers that survived all levels get forced into the last
        # level at whatever cell they fall in.
        if remaining:
            last = len(self.metadata_) - 1
            for cell, pt_idx in remaining.items():
                for i in pt_idx:
                    assignments[i] = SpatialAssignment(level=last, cell=cell)

        self._fitted = True
        return

    # ------------------------------------------------------------------
    # transform
    # ------------------------------------------------------------------

    def transform(
        self,
        points: np.ndarray,
        seed: Optional[int] = None,
    ) -> list[SpatialAssignment]:
        """
        Deterministically assign new points into the fitted hierarchy.

        Strategy: walk levels coarse-to-fine. At each level, for every
        cell, keep up to `limit` points (chosen by a seeded shuffle)
        and push the rest to the next finer level. Points still
        remaining after the last level are packed into their leaf cell.

        Parameters
        ----------
        points : (M, rank) array
        seed : int, optional
            RNG seed for the per-cell shuffle. Defaults to self.seed.

        Returns
        -------
        assignments : list[SpatialAssignment]
            Per-point level and cell.
        """
        self._ensure_fitted()

        points = np.asarray(points, dtype=np.float64)
        m = len(points)
        assignments: list[Optional[SpatialAssignment]] = [None] * m
        rng = np.random.default_rng(seed if seed is not None else self.seed)

        num_levels = len(self.metadata_)
        all_idx = np.arange(m, dtype=int)
        remaining = self._group_by_cell(all_idx, self._cell_of(points, 0))

        for level in range(num_levels):
            if not remaining:
                break

            leftovers: list[np.ndarray] = []

            for cell, pt_idx in remaining.items():
                if len(pt_idx) <= self.limit:
                    # Entire cell fits — emit everything here
                    for i in pt_idx:
                        assignments[i] = SpatialAssignment(level=level, cell=cell)
                else:
                    # Shuffle, take `limit`, push the rest down
                    rng.shuffle(pt_idx)
                    emitted, kept = pt_idx[: self.limit], pt_idx[self.limit :]
                    for i in emitted:
                        assignments[i] = SpatialAssignment(level=level, cell=cell)
                    leftovers.append(kept)

            # Re-bin leftovers into next level
            if level + 1 < num_levels and leftovers:
                leftover = np.concatenate(leftovers)
                remaining = self._group_by_cell(
                    leftover, self._cell_of(points[leftover], level + 1)
                )
            else:
                # Last level — force remaining into it
                if leftovers:
                    leftover = np.concatenate(leftovers)
                    coords = self._cell_of(points[leftover], level)
                    for i, coord in zip(leftover, coords):
                        assignments[i] = SpatialAssignment(
                            level=level, cell=tuple(coord)
                        )
                remaining = {}

        return assignments


# n_samples = 100_000
# sample_ratio = n_samples / len(synapses)
# target_limit = 5000
# input_limit = int(np.ceil(sample_ratio * target_limit))

# sit = SpatialIndexTree(limit=input_limit, max_levels=20)
# sit.fit(
#     synapses.sample(n_samples)[
#         ["ctr_pt_position_x", "ctr_pt_position_y", "ctr_pt_position_z"]
#     ].values
# )
# assignments = sit.transform(
#     synapses[["ctr_pt_position_x", "ctr_pt_position_y", "ctr_pt_position_z"]].values
# )
