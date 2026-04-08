"""Metadata (info JSON) generation for precomputed annotations."""

import json
from collections.abc import Sequence
from typing import Optional

import numpy as np
from neuroglancer import viewer_state
from neuroglancer.coordinate_space import CoordinateSpace

from ._sharding import ShardSpec
from ._spatial import SpatialLevel


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        return super().default(o)


def build_info(
    coordinate_space: CoordinateSpace,
    annotation_type: str,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    properties: Sequence[viewer_state.AnnotationPropertySpec],
    relationships: Sequence[str],
    levels: Sequence[SpatialLevel],
    by_id_sharding: Optional[ShardSpec] = None,
    spatial_shardings: Optional[dict[int, ShardSpec]] = None,
    relationship_shardings: Optional[dict[str, ShardSpec]] = None,
) -> dict:
    """Build the info JSON metadata for a precomputed annotation collection.

    Parameters
    ----------
    coordinate_space : CoordinateSpace
        Neuroglancer coordinate space with names, scales, units.
    annotation_type : str
        One of "point", "line", "axis_aligned_bounding_box", "ellipsoid".
    lower_bound : np.ndarray
        (rank,) lower bound of the annotation bounding box.
    upper_bound : np.ndarray
        (rank,) upper bound.
    properties : Sequence[AnnotationPropertySpec]
        Annotation properties.
    relationships : Sequence[str]
        Relationship names.
    levels : Sequence[SpatialLevel]
        Spatial index levels (coarsest first).
    by_id_sharding : ShardSpec or None
        Sharding spec for the by_id index.
    spatial_shardings : dict[int, ShardSpec] or None
        Per-level sharding specs, keyed by level index.
    relationship_shardings : dict[str, ShardSpec] or None
        Per-relationship sharding specs.

    Returns
    -------
    dict
        The info JSON metadata.
    """
    spatial = []
    for i, level in enumerate(levels):
        entry = {
            "key": level.key,
            "grid_shape": [int(x) for x in level.grid_shape],
            "chunk_size": [float(x) for x in level.chunk_size],
            "limit": level.limit,
        }
        if spatial_shardings and i in spatial_shardings:
            entry["sharding"] = spatial_shardings[i].to_json()
        spatial.append(entry)

    metadata = {
        "@type": "neuroglancer_annotations_v1",
        "dimensions": coordinate_space.to_json(),
        "lower_bound": [float(x) for x in lower_bound],
        "upper_bound": [float(x) for x in upper_bound],
        "annotation_type": annotation_type,
        "properties": [p.to_json() for p in properties],
        "relationships": [],
        "by_id": {"key": "by_id"},
        "spatial": spatial,
    }

    for relationship in relationships:
        rel_entry = {"id": relationship, "key": f"rel_{relationship}"}
        if relationship_shardings and relationship in relationship_shardings:
            rel_entry["sharding"] = relationship_shardings[relationship].to_json()
        metadata["relationships"].append(rel_entry)

    if by_id_sharding is not None:
        metadata["by_id"]["sharding"] = by_id_sharding.to_json()

    return metadata


def serialize_info(info: dict) -> str:
    """Serialize info dict to JSON string."""
    return json.dumps(info, cls=_NumpyEncoder)
