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

from ..statebuilder.utils import split_point_columns
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
    more DataFrames via the ``write()`` method. Currently only supports
    point annotations.

    Parameters
    ----------
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
    data_resolution : sequence of float, optional
        Resolution of the input coordinate data (e.g., ``[4, 4, 40]``).
        If provided, coordinates are rescaled from ``data_resolution``
        into the ``coordinate_space`` resolution before writing.
        Use this when your point coordinates are in a different voxel
        grid than the segmentation source (e.g., CAVE materialization
        data at viewer resolution vs. segmentation at native resolution).
    point_column : str or list of str
        Column(s) for point coordinates. Accepts:

        - A single column name containing ``[x, y, z]`` arrays
          (e.g., ``"pt_position"``).
        - A column name prefix that expands to ``{prefix}_x``,
          ``{prefix}_y``, ``{prefix}_z``
          (e.g., ``"ctr_pt_position"``).
        - An explicit list of three column names for x, y, z
          (e.g., ``["pt_x", "pt_y", "pt_z"]``).
    property_columns : list of str, optional
        Column names to include as annotation properties. Property types
        are auto-inferred from the DataFrame column dtypes. The column
        name is used as the property name in the output.
    relationship_columns : list of str, optional
        Column names to include as annotation relationships (linked
        segment IDs). Columns may contain scalar IDs or lists of IDs.
        The column name is used as the relationship name in the output.
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
            segmentation_source=client,
            point_column="pt_position",
            property_columns=["confidence"],
            relationship_columns=["pt_root_id"],
        )
        writer.write(df, "gs://bucket/annotations")

        # With split x/y/z columns via prefix
        writer = AnnotationDataFrameWriter(
            segmentation_source=client,
            point_column="ctr_pt_position",
            property_columns=["size"],
            relationship_columns=["pre_pt_root_id", "post_pt_root_id"],
            id_column="synapse_id",
        )
        writer.write(df, "gs://bucket/annotations")

        # With explicit x/y/z column names
        writer = AnnotationDataFrameWriter(
            resolution=[8, 8, 40],
            point_column=["pt_x", "pt_y", "pt_z"],
        )
        writer.write(df, "/local/path")
    """

    def __init__(
        self,
        segmentation_source=None,
        *,
        coordinate_space: Optional[CoordinateSpace] = None,
        resolution: Optional[Sequence[float]] = None,
        data_resolution: Optional[Sequence[float]] = None,
        point_column: Optional[Union[str, list[str]]] = None,
        property_columns: Optional[list[str]] = None,
        relationship_columns: Optional[list[str]] = None,
        id_column: Optional[str] = None,
        chunk_size: Optional[Union[float, Sequence[float]]] = None,
        limit: int = 5000,
        write_sharded: bool = True,
    ):
        self.annotation_type = "point"
        self.coordinate_space = resolve_coordinate_space(
            segmentation_source=segmentation_source,
            coordinate_space=coordinate_space,
            resolution=resolution,
        )
        if point_column is None:
            raise ValueError("point_column is required.")
        self.point_column = point_column
        self.property_columns = property_columns or []
        self.relationship_columns = relationship_columns or []
        self.id_column = id_column
        self.data_resolution = data_resolution
        self.chunk_size = chunk_size
        self.limit = limit
        self.write_sharded = write_sharded

    def _extract_coordinates(self, df: pd.DataFrame) -> np.ndarray:
        """Extract coordinate array from DataFrame."""
        resolved = split_point_columns(self.point_column, list(df.columns))
        if isinstance(resolved, list):
            x = _convert_arrow_column(df[resolved[0]])
            y = _convert_arrow_column(df[resolved[1]])
            z = _convert_arrow_column(df[resolved[2]])
            return np.column_stack([x, y, z]).astype(np.float32)
        else:
            col = df[resolved]
            return np.stack(col.values).astype(np.float32)

    def _resolve_properties(
        self, df: pd.DataFrame
    ) -> tuple[list[viewer_state.AnnotationPropertySpec], dict[str, np.ndarray]]:
        """Resolve property specs and extract property arrays from DataFrame."""
        specs = []
        arrays = {}

        for col_name in self.property_columns:
            prop_type = _infer_property_type(df[col_name])
            spec = viewer_state.AnnotationPropertySpec(id=col_name, type=prop_type)
            specs.append(spec)
            arrays[col_name] = _convert_arrow_column(df[col_name])

        return specs, arrays

    def _extract_relationships(
        self, df: pd.DataFrame
    ) -> tuple[list[str], dict[str, Union[np.ndarray, list]]]:
        """Extract relationship data from DataFrame."""
        names = []
        data = {}

        for col_name in self.relationship_columns:
            names.append(col_name)
            col = df[col_name]
            first_val = col.iloc[0] if len(col) > 0 else None

            if first_val is not None and isinstance(first_val, (list, np.ndarray)):
                # Variable-length: list of lists
                data[col_name] = [
                    list(v) if isinstance(v, (list, np.ndarray)) else [int(v)]
                    for v in col
                ]
            else:
                # Scalar column
                data[col_name] = _convert_arrow_column(col).astype(np.uint64)

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
            data_resolution=self.data_resolution,
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
