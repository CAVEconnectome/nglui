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

# Pre-computed uint type info for enum encoding (sorted smallest to largest)
_UINT_ENUM_TYPES: list[tuple[int, str, type]] = [
    (np.iinfo(np.uint8).max, "uint8", np.uint8),
    (np.iinfo(np.uint16).max, "uint16", np.uint16),
    (np.iinfo(np.uint32).max, "uint32", np.uint32),
]

# Maps pandas/numpy dtype kinds to neuroglancer property types
_DTYPE_TO_PROPERTY_TYPE = {
    "u": {1: "uint8", 2: "uint16", 4: "uint32"},
    "i": {1: "int8", 2: "int16", 4: "int32", 8: "int32"},
    "f": {2: "float32", 4: "float32", 8: "float32"},
}


def _choose_uint_for_count(n: int) -> tuple[str, type]:
    """Return the smallest uint neuroglancer type (str) and numpy dtype that fits n values."""
    for max_val, type_str, np_dtype in _UINT_ENUM_TYPES:
        if n <= max_val:
            return type_str, np_dtype
    raise ValueError(
        f"Too many distinct enum values ({n}) to encode in uint32 ({np.iinfo(np.uint32).max} max)."
    )


def _is_enum_column(series: pd.Series) -> bool:
    """Return True if this column should be encoded as a neuroglancer enum property."""
    dtype = series.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        return True
    if isinstance(dtype, pd.StringDtype):
        return True
    if isinstance(dtype, pd.BooleanDtype):
        return True
    if isinstance(dtype, np.dtype) and dtype.kind == "b":
        return True
    if dtype == object:
        first_non_null = next(
            (
                v
                for v in series
                if v is not None and not (isinstance(v, float) and np.isnan(v))
            ),
            None,
        )
        return isinstance(first_non_null, str)
    return False


def _build_enum_property(
    series: pd.Series, col_name: str
) -> tuple["viewer_state.AnnotationPropertySpec", np.ndarray]:
    """Build an enum AnnotationPropertySpec and encoded integer array for a string/categorical column."""
    has_nulls = series.isna().any()

    if isinstance(series.dtype, pd.BooleanDtype) or (
        isinstance(series.dtype, np.dtype) and series.dtype.kind == "b"
    ):
        # Bool: False=0, True=1; nullable adds "null" at 0 and shifts by +1
        if has_nulls:
            labels = ["null", "False", "True"]
            raw = series.to_numpy(dtype=object, na_value=None)
            codes = np.array(
                [0 if v is None else (2 if v else 1) for v in raw], dtype=np.int64
            )
        else:
            labels = ["False", "True"]
            codes = series.to_numpy(dtype=bool).astype(np.int64)
    elif isinstance(series.dtype, pd.CategoricalDtype):
        # Preserve existing category order
        categories = [str(c) for c in series.cat.categories]
        if has_nulls:
            labels = ["null"] + categories
            # cat.codes are 0-based; -1 means NaN — shift all by +1, NaN becomes 0
            codes = series.cat.codes.to_numpy().astype(np.int64) + 1
        else:
            labels = categories
            codes = series.cat.codes.to_numpy().astype(np.int64)
    else:
        # String or object dtype — use sorted unique non-null values
        non_null = series.dropna()
        unique_vals = sorted({str(v) for v in non_null})

        if has_nulls:
            labels = ["null"] + unique_vals
            label_to_code = {v: i + 1 for i, v in enumerate(unique_vals)}
        else:
            labels = unique_vals
            label_to_code = {v: i for i, v in enumerate(unique_vals)}

        def _encode(val):
            if val is None or (isinstance(val, float) and np.isnan(val)) or val != val:
                return 0
            return label_to_code.get(str(val), 0)

        if isinstance(series.dtype, pd.StringDtype):
            raw = series.to_numpy(dtype=object, na_value=None)
        else:
            raw = series.to_numpy()
        codes = np.array([_encode(v) for v in raw], dtype=np.int64)

    type_str, np_dtype = _choose_uint_for_count(len(labels))
    enum_values = list(range(len(labels)))
    spec = viewer_state.AnnotationPropertySpec(
        id=col_name,
        type=type_str,
        enum_values=enum_values,
        enum_labels=labels,
    )
    return spec, codes.astype(np_dtype)


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
            if _is_enum_column(df[col_name]):
                spec, arr = _build_enum_property(df[col_name], col_name)
            else:
                prop_type = _infer_property_type(df[col_name])
                spec = viewer_state.AnnotationPropertySpec(id=col_name, type=prop_type)
                arr = _convert_arrow_column(df[col_name])
            specs.append(spec)
            arrays[col_name] = arr

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
        print("Extracting coords...")
        coords = self._extract_coordinates(df)
        print("Resolving properties...")
        properties, prop_arrays = self._resolve_properties(df)
        print("Extracting relationships...")
        rel_names, rel_data = self._extract_relationships(df)
        print("Extracting IDs...")
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

        print("Setting coords in writer...")
        writer.set_coordinates(coords)
        print("Setting IDs in writer...")
        writer.set_ids(ids)

        for prop_name, arr in prop_arrays.items():
            writer.set_property(prop_name, arr)

        for rel_name, data in rel_data.items():
            writer.set_relationship(rel_name, data)

        print("Writing precomputed data...")
        writer.write(path)
