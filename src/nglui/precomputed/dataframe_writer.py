"""High-level DataFrame-to-precomputed annotation writer.

Maps pandas DataFrame columns to neuroglancer precomputed annotation format
via the bulk array API of PrecomputedAnnotationWriter.
"""

from collections.abc import Sequence
from typing import Optional, Union

import numpy as np
import pandas as pd
from neuroglancer import viewer_state
from neuroglancer.coordinate_space import CoordinateSpace

from ._encoding import AnnotationType
from ._source import resolve_coordinate_space
from .writer import PrecomputedAnnotationWriter

# Maps pandas/numpy dtype kinds to neuroglancer property types
_DTYPE_TO_PROPERTY_TYPE = {
    "u": {1: "uint8", 2: "uint16", 4: "uint32"},
    "i": {1: "int8", 2: "int16", 4: "int32", 8: "int32"},
    "f": {2: "float32", 4: "float32", 8: "float32"},
}


def _infer_property_type(series: pd.Series) -> str:
    """Infer neuroglancer property type from a pandas Series dtype."""
    dtype = series.dtype
    if hasattr(dtype, "numpy_dtype"):
        # ArrowDtype
        dtype = dtype.numpy_dtype

    kind = np.dtype(dtype).kind
    itemsize = np.dtype(dtype).itemsize

    if kind in _DTYPE_TO_PROPERTY_TYPE:
        type_map = _DTYPE_TO_PROPERTY_TYPE[kind]
        if itemsize in type_map:
            return type_map[itemsize]
        # Fall back to largest available
        return type_map[max(type_map)]

    raise ValueError(
        f"Cannot infer neuroglancer property type from dtype {dtype}. "
        f"Provide an explicit AnnotationPropertySpec."
    )


def _convert_arrow_column(series: pd.Series) -> np.ndarray:
    """Convert a potentially PyArrow-backed column to numpy."""
    if hasattr(series.dtype, "numpy_dtype"):
        return series.to_numpy(dtype=series.dtype.numpy_dtype, na_value=0)
    if isinstance(series.dtype, pd.StringDtype):
        return series.to_numpy(dtype=object, na_value="")
    return series.to_numpy()


class AnnotationDataFrameWriter:
    """Write pandas DataFrames as neuroglancer precomputed annotations.

    Configures column mappings at construction time, then writes one or
    more DataFrames via the ``write()`` method.

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
    point_column : str, optional
        Column containing [x, y, z] arrays (for point annotations).
    x_column, y_column, z_column : str, optional
        Separate columns for x, y, z coordinates (for point annotations).
    point_a_column, point_b_column : str, optional
        Columns for two-point annotations (line, bbox).
    center_column, radii_column : str, optional
        Columns for ellipsoid annotations.
    property_columns : dict, optional
        Maps annotation property name to DataFrame column name or
        AnnotationPropertySpec. If a string, the property type is
        auto-inferred from the DataFrame column dtype.
    relationship_columns : dict, optional
        Maps relationship name to DataFrame column name.
        Column may contain scalar IDs or lists of IDs.
    id_column : str, optional
        Column for annotation IDs. If None, uses DataFrame index.
    chunk_size : float or array-like, optional
        Spatial index chunk size. If None, auto-computed.
    limit : int
        Target max annotations per spatial chunk (default 5000).
    write_sharded : bool
        Use sharded writes (default True).

    Examples
    --------
    ::

        # From a CAVEclient (most common)
        writer = AnnotationDataFrameWriter(
            annotation_type="point",
            segmentation_source=client,
            point_column="pt_position",
            property_columns={"score": "confidence"},
            relationship_columns={"segment": "pt_root_id"},
        )
        writer.write(df, "gs://bucket/annotations")

        # With explicit resolution
        writer = AnnotationDataFrameWriter(
            annotation_type="point",
            resolution=[8, 8, 40],
            x_column="x", y_column="y", z_column="z",
        )
        writer.write(df, "/local/path")
    """

    def __init__(
        self,
        annotation_type: AnnotationType,
        segmentation_source=None,
        *,
        coordinate_space: Optional[CoordinateSpace] = None,
        resolution: Optional[Sequence[float]] = None,
        point_column: Optional[str] = None,
        x_column: Optional[str] = None,
        y_column: Optional[str] = None,
        z_column: Optional[str] = None,
        point_a_column: Optional[str] = None,
        point_b_column: Optional[str] = None,
        center_column: Optional[str] = None,
        radii_column: Optional[str] = None,
        property_columns: Optional[
            dict[str, Union[str, viewer_state.AnnotationPropertySpec]]
        ] = None,
        relationship_columns: Optional[dict[str, str]] = None,
        id_column: Optional[str] = None,
        chunk_size: Optional[Union[float, Sequence[float]]] = None,
        limit: int = 5000,
        write_sharded: bool = True,
    ):
        self.annotation_type = annotation_type
        self.coordinate_space = resolve_coordinate_space(
            segmentation_source=segmentation_source,
            coordinate_space=coordinate_space,
            resolution=resolution,
        )
        self.point_column = point_column
        self.x_column = x_column
        self.y_column = y_column
        self.z_column = z_column
        self.point_a_column = point_a_column
        self.point_b_column = point_b_column
        self.center_column = center_column
        self.radii_column = radii_column
        self.property_columns = property_columns or {}
        self.relationship_columns = relationship_columns or {}
        self.id_column = id_column
        self.chunk_size = chunk_size
        self.limit = limit
        self.write_sharded = write_sharded

        self._validate_column_config()

    def _validate_column_config(self):
        """Validate that column configuration is consistent."""
        if self.annotation_type == "point":
            has_array = self.point_column is not None
            has_xyz = all(
                c is not None for c in [self.x_column, self.y_column, self.z_column]
            )
            if not has_array and not has_xyz:
                raise ValueError(
                    "Point annotations require either point_column or "
                    "x_column/y_column/z_column."
                )
            if has_array and has_xyz:
                raise ValueError(
                    "Specify either point_column or x/y/z columns, not both."
                )
        elif self.annotation_type in ("line", "axis_aligned_bounding_box"):
            if self.point_a_column is None or self.point_b_column is None:
                raise ValueError(
                    f"{self.annotation_type} requires point_a_column and point_b_column."
                )
        elif self.annotation_type == "ellipsoid":
            if self.center_column is None or self.radii_column is None:
                raise ValueError(
                    "Ellipsoid annotations require center_column and radii_column."
                )

    def _extract_coordinates(self, df: pd.DataFrame) -> np.ndarray:
        """Extract coordinate array from DataFrame."""
        if self.annotation_type == "point":
            if self.point_column is not None:
                col = df[self.point_column]
                return np.stack(col.values).astype(np.float32)
            else:
                x = _convert_arrow_column(df[self.x_column])
                y = _convert_arrow_column(df[self.y_column])
                z = _convert_arrow_column(df[self.z_column])
                return np.column_stack([x, y, z]).astype(np.float32)

        elif self.annotation_type in ("line", "axis_aligned_bounding_box"):
            a = np.stack(df[self.point_a_column].values).astype(np.float32)
            b = np.stack(df[self.point_b_column].values).astype(np.float32)
            return np.concatenate([a, b], axis=1)

        elif self.annotation_type == "ellipsoid":
            center = np.stack(df[self.center_column].values).astype(np.float32)
            radii = np.stack(df[self.radii_column].values).astype(np.float32)
            return np.concatenate([center, radii], axis=1)

        raise ValueError(f"Unsupported annotation type: {self.annotation_type}")

    def _resolve_properties(
        self, df: pd.DataFrame
    ) -> tuple[list[viewer_state.AnnotationPropertySpec], dict[str, np.ndarray]]:
        """Resolve property specs and extract property arrays from DataFrame."""
        specs = []
        arrays = {}

        for prop_name, col_spec in self.property_columns.items():
            if isinstance(col_spec, viewer_state.AnnotationPropertySpec):
                spec = col_spec
                col_name = spec.id
            else:
                col_name = col_spec
                prop_type = _infer_property_type(df[col_name])
                spec = viewer_state.AnnotationPropertySpec(id=prop_name, type=prop_type)

            specs.append(spec)
            arrays[prop_name] = _convert_arrow_column(df[col_name])

        return specs, arrays

    def _extract_relationships(
        self, df: pd.DataFrame
    ) -> tuple[list[str], dict[str, Union[np.ndarray, list]]]:
        """Extract relationship data from DataFrame."""
        names = []
        data = {}

        for rel_name, col_name in self.relationship_columns.items():
            names.append(rel_name)
            col = df[col_name]
            first_val = col.iloc[0] if len(col) > 0 else None

            if first_val is not None and isinstance(first_val, (list, np.ndarray)):
                # Variable-length: list of lists
                data[rel_name] = [
                    list(v) if isinstance(v, (list, np.ndarray)) else [int(v)]
                    for v in col
                ]
            else:
                # Scalar column
                data[rel_name] = _convert_arrow_column(col).astype(np.uint64)

        return names, data

    def _extract_ids(self, df: pd.DataFrame) -> np.ndarray:
        """Extract annotation IDs from DataFrame."""
        if self.id_column is not None:
            return _convert_arrow_column(df[self.id_column]).astype(np.uint64)
        else:
            return np.asarray(df.index, dtype=np.uint64)

    def write(self, df: pd.DataFrame, path: str):
        """Write a DataFrame as precomputed annotations.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing annotation data.
        path : str
            Output path (local or cloud).
        """
        coords = self._extract_coordinates(df)
        properties, prop_arrays = self._resolve_properties(df)
        rel_names, rel_data = self._extract_relationships(df)
        ids = self._extract_ids(df)

        writer = PrecomputedAnnotationWriter(
            annotation_type=self.annotation_type,
            coordinate_space=self.coordinate_space,
            relationships=rel_names,
            properties=properties,
            chunk_size=self.chunk_size,
            limit=self.limit,
            write_sharded=self.write_sharded,
        )

        writer.set_coordinates(coords)
        writer.set_ids(ids)

        for prop_name, arr in prop_arrays.items():
            writer.set_property(prop_name, arr)

        for rel_name, data in rel_data.items():
            writer.set_relationship(rel_name, data)

        writer.write(path)
