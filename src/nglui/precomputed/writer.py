"""Low-level precomputed annotation writer.

Provides both a bulk array API (fast, for use by DataFrame writer)
and a per-row API (streaming, for future polyline support).
Writes to local or cloud storage via tensorstore.
"""

import numbers
import os
import struct
from collections import defaultdict
from collections.abc import Sequence
from typing import Literal, Optional, Union

import numpy as np
from neuroglancer import viewer_state
from neuroglancer.coordinate_space import CoordinateSpace

from ._encoding import (
    AnnotationType,
    build_dtype,
    encode_by_id_entries,
    encode_fixed_blocks,
    encode_multiple_annotations,
    sort_properties,
)
from ._metadata import build_info, serialize_info
from ._sharding import ShardSpec, choose_output_spec
from ._source import resolve_coordinate_space
from ._spatial import (
    SpatialLevel,
    auto_chunk_size,
    build_spatial_levels,
    compressed_morton_code,
    compute_multiscale_assignments,
    encode_multiscale_spatial_chunks,
)

try:
    import tensorstore as ts

    _has_tensorstore = True
except ImportError:
    _has_tensorstore = False


def _ensure_tensorstore():
    if not _has_tensorstore:
        raise ImportError(
            "tensorstore is required for precomputed annotation writing. "
            "Install with: pip install 'nglui[precomputed]'"
        )


class PrecomputedAnnotationWriter:
    """Write annotations in the neuroglancer precomputed format.

    Supports two usage patterns:

    **Bulk array API** (fast, used by AnnotationDataFrameWriter)::

        writer = PrecomputedAnnotationWriter(
            annotation_type="point",
            segmentation_source=client,
        )
        writer.set_coordinates(coords_array)
        writer.set_property("score", scores_array)
        writer.set_relationship("segment", segment_ids)
        writer.write("/path/to/output")

    **Per-row API** (streaming)::

        writer = PrecomputedAnnotationWriter(
            annotation_type="point",
            resolution=[8, 8, 40],
            relationships=["segment"],
            properties=[AnnotationPropertySpec(id="score", type="float32")],
        )
        writer.add_point([100, 200, 50], score=0.95, segment=123)
        writer.add_point([110, 210, 55], score=0.87, segment=456)
        writer.write("/path/to/output")

    Parameters
    ----------
    annotation_type : str
        One of "point", "line", "axis_aligned_bounding_box", "ellipsoid".
    segmentation_source : CAVEclient, CloudVolume, or str, optional
        Source to derive coordinate space from. Accepts a CAVEclient
        (uses its segmentation source), a CloudVolume instance, or a
        segmentation source URL. Mutually exclusive with
        ``coordinate_space`` and ``resolution``.
    coordinate_space : CoordinateSpace, optional
        Explicit neuroglancer coordinate space. Mutually exclusive with
        ``segmentation_source`` and ``resolution``.
    resolution : sequence of float, optional
        Explicit resolution per axis (e.g., ``[8, 8, 40]``). Units
        default to nm, axis names to x/y/z. Mutually exclusive with
        ``segmentation_source`` and ``coordinate_space``.
    relationships : Sequence[str], optional
        Names of relationships (e.g., segment ID links).
    properties : Sequence[AnnotationPropertySpec], optional
        Annotation property definitions.
    chunk_size : float or array-like, optional
        Spatial index chunk size for the finest level. If None,
        auto-computed from data bounds.
    limit : int
        Target max annotations per spatial chunk (default 5000).
        Controls both the finest grid size and the probabilistic
        subsampling at coarser levels.
    spatial_levels : list[SpatialLevel], optional
        Explicit spatial level hierarchy. If provided, overrides
        auto-computation from ``chunk_size`` and ``limit``.
    write_sharded : bool
        Whether to use sharded writes (default True).
    """

    def __init__(
        self,
        annotation_type: AnnotationType,
        segmentation_source=None,
        *,
        coordinate_space: Optional[CoordinateSpace] = None,
        resolution: Optional[Sequence[float]] = None,
        relationships: Sequence[str] = (),
        properties: Sequence[viewer_state.AnnotationPropertySpec] = (),
        chunk_size: Optional[Union[float, Sequence[float]]] = None,
        limit: int = 5000,
        spatial_levels: Optional[list[SpatialLevel]] = None,
        write_sharded: bool = True,
    ):
        self.coordinate_space = resolve_coordinate_space(
            segmentation_source=segmentation_source,
            coordinate_space=coordinate_space,
            resolution=resolution,
        )
        self.annotation_type = annotation_type
        self.rank = self.coordinate_space.rank
        self.relationships = list(relationships)
        self.properties = sort_properties(properties)
        self.write_sharded = write_sharded
        self.limit = limit
        self._spatial_levels = spatial_levels

        self._dtype = build_dtype(annotation_type, self.rank, self.properties)

        # Chunk size: store raw user input, resolve at write time
        self._user_chunk_size = chunk_size

        # Bulk data (set via set_* methods or accumulated via add_* methods)
        self._coordinates: Optional[np.ndarray] = None
        self._properties: dict[str, np.ndarray] = {}
        self._relationships: dict[str, Union[np.ndarray, list]] = {}
        self._ids: Optional[np.ndarray] = None

        # Per-row accumulation buffers (used by add_* methods)
        self._row_coords: list[np.ndarray] = []
        self._row_properties: dict[str, list] = defaultdict(list)
        self._row_relationships: dict[str, list] = defaultdict(list)
        self._row_ids: list[int] = []

    # ── Bulk array API ──────────────────────────────────────────────────

    def set_coordinates(self, coordinates: np.ndarray):
        """Set all annotation coordinates at once.

        Parameters
        ----------
        coordinates : np.ndarray
            (N, rank) for points, (N, 2*rank) for line/bbox/ellipsoid.
        """
        self._coordinates = np.asarray(coordinates, dtype=np.float32)

    def set_property(self, name: str, values: np.ndarray):
        """Set values for a named property.

        Parameters
        ----------
        name : str
            Property id (must match one of the property specs).
        values : np.ndarray
            (N,) array of property values.
        """
        self._properties[name] = np.asarray(values)

    def set_relationship(
        self, name: str, ids: Union[np.ndarray, list[list[int]], list[int]]
    ):
        """Set relationship IDs for a named relationship.

        Parameters
        ----------
        name : str
            Relationship name.
        ids : np.ndarray or list
            (N,) array of scalar segment IDs, or list of lists for variable-length.
        """
        self._relationships[name] = ids

    def set_ids(self, ids: np.ndarray):
        """Set annotation IDs.

        Parameters
        ----------
        ids : np.ndarray
            (N,) uint64 array of annotation IDs.
        """
        self._ids = np.asarray(ids, dtype=np.uint64)

    # ── Per-row API ─────────────────────────────────────────────────────

    def add_point(self, point: Sequence[float], id: Optional[int] = None, **kwargs):
        """Add a single point annotation.

        Parameters
        ----------
        point : Sequence[float]
            (rank,) coordinate.
        id : int, optional
            Annotation ID. Auto-assigned if None.
        **kwargs
            Property values and relationship IDs by name.
        """
        if self.annotation_type != "point":
            raise ValueError(f"Cannot add point to {self.annotation_type} writer")
        self._add_row(np.asarray(point, dtype=np.float32), id, **kwargs)

    def add_line(
        self,
        point_a: Sequence[float],
        point_b: Sequence[float],
        id: Optional[int] = None,
        **kwargs,
    ):
        """Add a single line annotation."""
        if self.annotation_type != "line":
            raise ValueError(f"Cannot add line to {self.annotation_type} writer")
        coords = np.concatenate(
            [
                np.asarray(point_a, dtype=np.float32),
                np.asarray(point_b, dtype=np.float32),
            ]
        )
        self._add_row(coords, id, **kwargs)

    def add_ellipsoid(
        self,
        center: Sequence[float],
        radii: Sequence[float],
        id: Optional[int] = None,
        **kwargs,
    ):
        """Add a single ellipsoid annotation."""
        if self.annotation_type != "ellipsoid":
            raise ValueError(f"Cannot add ellipsoid to {self.annotation_type} writer")
        coords = np.concatenate(
            [np.asarray(center, dtype=np.float32), np.asarray(radii, dtype=np.float32)]
        )
        self._add_row(coords, id, **kwargs)

    def add_axis_aligned_bounding_box(
        self,
        point_a: Sequence[float],
        point_b: Sequence[float],
        id: Optional[int] = None,
        **kwargs,
    ):
        """Add a single axis-aligned bounding box annotation."""
        if self.annotation_type != "axis_aligned_bounding_box":
            raise ValueError(
                f"Cannot add bounding box to {self.annotation_type} writer"
            )
        coords = np.concatenate(
            [
                np.asarray(point_a, dtype=np.float32),
                np.asarray(point_b, dtype=np.float32),
            ]
        )
        self._add_row(coords, id, **kwargs)

    def _add_row(self, coords: np.ndarray, id: Optional[int], **kwargs):
        """Add a single annotation from the per-row API."""
        self._row_coords.append(coords)
        if id is None:
            id = len(self._row_coords) - 1
        self._row_ids.append(id)

        for p in self.properties:
            value = kwargs.pop(p.id, 0)
            self._row_properties[p.id].append(value)

        for rel in self.relationships:
            ids = kwargs.pop(rel, [])
            if isinstance(ids, numbers.Integral):
                ids = [ids]
            self._row_relationships[rel].append(ids)

        if kwargs:
            raise ValueError(f"Unexpected keyword arguments: {kwargs}")

    # ── Write ───────────────────────────────────────────────────────────

    def _finalize_data(self):
        """Merge per-row data into bulk arrays if needed."""
        if self._coordinates is not None:
            # Bulk API was used; data is already set
            coords = self._coordinates
        elif self._row_coords:
            # Per-row API was used; stack into arrays
            coords = np.stack(self._row_coords).astype(np.float32)
            for name, values in self._row_properties.items():
                if name not in self._properties:
                    self._properties[name] = np.asarray(values)
            for name, values in self._row_relationships.items():
                if name not in self._relationships:
                    self._relationships[name] = values
            if self._ids is None and self._row_ids:
                self._ids = np.array(self._row_ids, dtype=np.uint64)
        else:
            raise ValueError("No annotation data provided.")

        n = len(coords)
        if self._ids is None:
            self._ids = np.arange(n, dtype=np.uint64)

        return coords, n

    def _resolve_relationships(
        self, n: int
    ) -> tuple[list[list[list[int]]] | None, dict[str, dict[int, list[int]]]]:
        """Convert relationship data to canonical formats.

        Returns
        -------
        by_id_rels : list[list[list[int]]] or None
            relationships[r][i] = list of segment IDs for relationship r, annotation i.
        inverted_rels : dict[str, dict[int, list[int]]]
            Maps relationship name → {segment_id: [annotation_indices]}.
        """
        if not self.relationships:
            return None, {}

        by_id_rels: list[list[list[int]]] = []
        inverted_rels: dict[str, dict[int, list[int]]] = {}

        for rel_name in self.relationships:
            raw = self._relationships.get(rel_name)
            rel_per_row: list[list[int]] = []
            inverted: dict[int, list[int]] = defaultdict(list)

            if raw is None:
                # No data for this relationship
                rel_per_row = [[] for _ in range(n)]
            elif isinstance(raw, np.ndarray) and raw.ndim == 1:
                # Scalar: one ID per row
                for i, sid in enumerate(raw):
                    sid_int = int(sid)
                    if sid_int != 0:
                        rel_per_row.append([sid_int])
                        inverted[sid_int].append(i)
                    else:
                        rel_per_row.append([])
            elif isinstance(raw, list):
                # List of lists: variable IDs per row
                for i, ids in enumerate(raw):
                    if isinstance(ids, numbers.Integral):
                        ids = [ids]
                    ids_int = [int(x) for x in ids]
                    rel_per_row.append(ids_int)
                    for sid in ids_int:
                        inverted[sid].append(i)
            else:
                raise ValueError(
                    f"Unexpected relationship data type for '{rel_name}': {type(raw)}"
                )

            by_id_rels.append(rel_per_row)
            inverted_rels[rel_name] = dict(inverted)

        return by_id_rels, inverted_rels

    def write(self, path: str):
        """Write all annotations to the given path.

        Parameters
        ----------
        path : str
            Output path. Can be local filesystem path, or a cloud path
            (gs://, s3://) when using sharded writes with tensorstore.
        """
        coords, n = self._finalize_data()

        # Compute bounds
        if self.annotation_type == "point":
            lower_bound = coords.min(axis=0)
            upper_bound = coords.max(axis=0)
        else:
            coords_a = coords[:, : self.rank]
            coords_b = coords[:, self.rank :]
            lower_bound = np.minimum(coords_a, coords_b).min(axis=0)
            upper_bound = np.maximum(coords_a, coords_b).max(axis=0)

        # Small epsilon to avoid zero-width bounds and float32 precision issues
        extent = upper_bound - lower_bound
        extent = np.maximum(extent, 1.0)
        upper_bound = lower_bound + extent
        upper_bound = np.nextafter(upper_bound, np.inf).astype(np.float64)

        # Build spatial level hierarchy
        if self._spatial_levels is not None:
            levels = self._spatial_levels
        elif self._user_chunk_size is not None:
            # User specified chunk_size → use as finest level, coarsen from there
            if isinstance(self._user_chunk_size, numbers.Real):
                finest_cs = np.full(self.rank, float(self._user_chunk_size))
            else:
                finest_cs = np.asarray(self._user_chunk_size, dtype=np.float64)
            finest_grid = np.maximum(np.ceil(extent / finest_cs).astype(int), 1)
            # Snap upper_bound to grid
            upper_bound = lower_bound + finest_grid * finest_cs

            # Build coarser levels from finest
            grids = [finest_grid]
            while np.any(grids[-1] > 1):
                coarser = np.maximum(np.ceil(grids[-1] / 2).astype(int), 1)
                grids.append(coarser)
            grids.reverse()

            levels = []
            for i, grid in enumerate(grids):
                cs = (upper_bound - lower_bound) / grid
                levels.append(
                    SpatialLevel(
                        key=f"spatial{i}",
                        grid_shape=grid,
                        chunk_size=cs,
                        limit=self.limit,
                    )
                )
        else:
            levels = build_spatial_levels(lower_bound, upper_bound, n, limit=self.limit)
            # Snap upper_bound to finest grid
            finest = levels[-1]
            upper_bound = lower_bound + finest.grid_shape * finest.chunk_size

        # Compute multi-scale assignments
        assignment = compute_multiscale_assignments(coords, lower_bound, levels)

        # Encode fixed blocks
        fixed_blocks = encode_fixed_blocks(coords, self._properties, self._dtype)
        ids = self._ids

        # Resolve relationships
        by_id_rels, inverted_rels = self._resolve_relationships(n)

        # Encode by_id entries
        by_id_entries = encode_by_id_entries(fixed_blocks, by_id_rels)

        # Encode multi-scale spatial chunks
        spatial_chunks = encode_multiscale_spatial_chunks(fixed_blocks, ids, assignment)

        # Compute sharding specs
        total_ann_bytes = sum(len(e) for e in by_id_entries)

        by_id_sharding = None
        spatial_shardings: dict[int, ShardSpec] = {}
        relationship_shardings = {}
        if self.write_sharded and _has_tensorstore:
            by_id_sharding = choose_output_spec(n, total_ann_bytes)
            for rel_name, inv in inverted_rels.items():
                rel_total = sum(
                    len(
                        encode_multiple_annotations(
                            fixed_blocks[np.array(idx)], ids[np.array(idx)]
                        )
                    )
                    for idx in inv.values()
                )
                rel_shard = choose_output_spec(len(inv), rel_total)
                if rel_shard is not None:
                    relationship_shardings[rel_name] = rel_shard

        # Build metadata
        info = build_info(
            coordinate_space=self.coordinate_space,
            annotation_type=self.annotation_type,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            properties=self.properties,
            relationships=self.relationships,
            levels=levels,
            by_id_sharding=by_id_sharding,
            spatial_shardings=spatial_shardings or None,
            relationship_shardings=relationship_shardings or None,
        )

        # Write everything
        if self.write_sharded and _has_tensorstore and by_id_sharding is not None:
            self._write_sharded(
                path,
                info,
                by_id_entries,
                ids,
                spatial_chunks,
                levels,
                by_id_sharding,
                inverted_rels,
                relationship_shardings,
                fixed_blocks,
            )
        else:
            self._write_unsharded(
                path,
                info,
                by_id_entries,
                ids,
                spatial_chunks,
                levels,
                inverted_rels,
                fixed_blocks,
            )

    def _write_unsharded(
        self,
        path,
        info,
        by_id_entries,
        ids,
        spatial_chunks,
        levels,
        inverted_rels,
        fixed_blocks,
    ):
        """Write annotation data in unsharded format to local filesystem."""
        os.makedirs(path, exist_ok=True)
        os.makedirs(os.path.join(path, "by_id"), exist_ok=True)
        for level in levels:
            os.makedirs(os.path.join(path, level.key), exist_ok=True)
        for rel_name in self.relationships:
            os.makedirs(os.path.join(path, f"rel_{rel_name}"), exist_ok=True)

        # Write info
        with open(os.path.join(path, "info"), "w") as f:
            f.write(serialize_info(info))

        # Write by_id
        for i, entry in enumerate(by_id_entries):
            with open(os.path.join(path, "by_id", str(int(ids[i]))), "wb") as f:
                f.write(entry)

        # Write spatial chunks (multi-level)
        for (level_idx, cell), data in spatial_chunks.items():
            level_key = levels[level_idx].key
            chunk_name = "_".join(str(c) for c in cell)
            with open(os.path.join(path, level_key, chunk_name), "wb") as f:
                f.write(data)

        # Write relationship indexes
        for rel_name, inv in inverted_rels.items():
            for segment_id, ann_indices in inv.items():
                idx_arr = np.array(ann_indices)
                chunk_data = encode_multiple_annotations(
                    fixed_blocks[idx_arr], ids[idx_arr]
                )
                filepath = os.path.join(path, f"rel_{rel_name}", str(segment_id))
                with open(filepath, "wb") as f:
                    f.write(chunk_data)

    def _write_sharded(
        self,
        path,
        info,
        by_id_entries,
        ids,
        spatial_chunks,
        levels,
        by_id_sharding,
        inverted_rels,
        relationship_shardings,
        fixed_blocks,
    ):
        """Write annotation data in sharded format via tensorstore."""
        _ensure_tensorstore()

        # Ensure path has file:// for local paths
        if not path.startswith(("gs://", "s3://", "file://", "http://", "https://")):
            ts_base = f"file://{os.path.abspath(path)}"
        else:
            ts_base = path

        os.makedirs(path, exist_ok=True) if not path.startswith(
            ("gs://", "s3://")
        ) else None

        # Write info via tensorstore json driver
        info_spec = {"driver": "json", "kvstore": os.path.join(ts_base, "info")}
        info_store = ts.open(info_spec).result()
        info_store.write(info).result()

        # Write by_id (sharded)
        by_id_spec = {
            "driver": "neuroglancer_uint64_sharded",
            "metadata": by_id_sharding.to_json(),
            "base": os.path.join(ts_base, "by_id"),
        }
        by_id_store = ts.KvStore.open(by_id_spec).result()
        txn = ts.Transaction()
        for i, entry in enumerate(by_id_entries):
            key = np.ascontiguousarray(ids[i], dtype=">u8").tobytes()
            by_id_store.with_transaction(txn)[key] = entry
        txn.commit_async().result()

        # Write spatial chunks per level (unsharded via tensorstore KvStore)
        for (level_idx, cell), data in spatial_chunks.items():
            level_key = levels[level_idx].key
            spatial_spec = os.path.join(ts_base, level_key) + "/"
            spatial_store = ts.KvStore.open(spatial_spec).result()
            chunk_name = "_".join(str(c) for c in cell)
            spatial_store[chunk_name] = data

        # Write relationship indexes
        for rel_name, inv in inverted_rels.items():
            rel_sharding = relationship_shardings.get(rel_name)
            if rel_sharding is not None:
                rel_spec = {
                    "driver": "neuroglancer_uint64_sharded",
                    "metadata": rel_sharding.to_json(),
                    "base": os.path.join(ts_base, f"rel_{rel_name}"),
                }
                rel_store = ts.KvStore.open(rel_spec).result()
                txn = ts.Transaction()
                for segment_id, ann_indices in inv.items():
                    key = np.ascontiguousarray(segment_id, dtype=">u8").tobytes()
                    idx_arr = np.array(ann_indices)
                    value = encode_multiple_annotations(
                        fixed_blocks[idx_arr], ids[idx_arr]
                    )
                    rel_store.with_transaction(txn)[key] = value
                txn.commit_async().result()
            else:
                rel_spec = os.path.join(ts_base, f"rel_{rel_name}") + "/"
                rel_store = ts.KvStore.open(rel_spec).result()
                for segment_id, ann_indices in inv.items():
                    idx_arr = np.array(ann_indices)
                    value = encode_multiple_annotations(
                        fixed_blocks[idx_arr], ids[idx_arr]
                    )
                    rel_store[str(segment_id)] = value
