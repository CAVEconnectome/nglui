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
from ._spatial import _IsotropicHierarchy, _UniformHierarchy
from ._writer import _PrecomputedAnnotationWriter

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


class _AnnotationWriter:
    """Private base class for DataFrame-to-precomputed annotation writers.

    Handles all annotation geometry types. Public subclasses expose only
    the coordinate column parameters relevant to their geometry.

    Parameters
    ----------
    annotation_type : AnnotationType
        The annotation geometry type.
    coordinate_columns : list of (str, str or list of str)
        Each entry is ``(label, column_spec)`` where *column_spec* is
        resolved via ``split_point_columns`` to extract one position
        vector per annotation. Results are concatenated column-wise
        to form the full coordinate array.
    segmentation_source : CAVEclient, CloudVolume, or str, optional
        Source to derive coordinate space from.
    coordinate_space : CoordinateSpace, optional
        Explicit neuroglancer coordinate space.
    resolution : sequence of float, optional
        Explicit resolution per axis.
    data_resolution : sequence of float, optional
        Resolution of the input coordinate data for rescaling.
    property_columns : str or list of str, optional
        Column name(s) to include as annotation properties.
    relationship_columns : str or list of str, optional
        Column name(s) to include as annotation relationships.
    id_column : str, optional
        Column for annotation IDs. If None, uses DataFrame index.
    chunk_size : float or array-like, optional
        Spatial index chunk size. If set, uses uniform-grid spatial indexing.
    limit : int
        Target max annotations per spatial chunk.
    write_sharded : bool
        Use sharded writes (default True).
    """

    def __init__(
        self,
        annotation_type: AnnotationType,
        coordinate_columns: list[tuple[str, Union[str, list[str]]]],
        segmentation_source=None,
        coordinate_space: Optional[CoordinateSpace] = None,
        resolution: Optional[Sequence[float]] = None,
        data_resolution: Optional[Sequence[float]] = None,
        property_columns: Optional[Union[str, list[str]]] = None,
        relationship_columns: Optional[Union[str, list[str]]] = None,
        id_column: Optional[str] = None,
        chunk_size: Optional[Union[float, Sequence[float]]] = None,
        limit: int = 10_000,
    ):
        self.annotation_type = annotation_type
        self._coordinate_columns = coordinate_columns
        self.coordinate_space = resolve_coordinate_space(
            segmentation_source=segmentation_source,
            coordinate_space=coordinate_space,
            resolution=resolution,
        )
        # Normalize property_columns to always be a list
        if isinstance(property_columns, str):
            self.property_columns = [property_columns]
        else:
            self.property_columns = property_columns or []
        # Normalize relationship_columns to always be a list
        if isinstance(relationship_columns, str):
            self.relationship_columns = [relationship_columns]
        else:
            self.relationship_columns = relationship_columns or []
        self.id_column = id_column
        self.data_resolution = data_resolution

        # Resolve spatial hierarchy from convenience params
        rank = self.coordinate_space.rank
        if chunk_size is not None:
            self.spatial_hierarchy = _UniformHierarchy(
                rank=rank, chunk_size=chunk_size, limit=limit
            )
        else:
            self.spatial_hierarchy = _IsotropicHierarchy(rank=rank, limit=limit)

    def _extract_one_position(
        self, df: pd.DataFrame, column_spec: Union[str, list[str]]
    ) -> np.ndarray:
        """Extract one (N, 3) position array from a column spec."""
        resolved = split_point_columns(column_spec, list(df.columns))
        if isinstance(resolved, list):
            x = _convert_arrow_column(df[resolved[0]])
            y = _convert_arrow_column(df[resolved[1]])
            z = _convert_arrow_column(df[resolved[2]])
            return np.column_stack([x, y, z]).astype(np.float32)
        else:
            col = df[resolved]
            return np.stack(col.values).astype(np.float32)

    def _extract_coordinates(self, df: pd.DataFrame) -> np.ndarray:
        """Extract coordinate array from DataFrame.

        Resolves each coordinate column spec and concatenates them
        column-wise. Returns (N, rank) for point, (N, 2*rank) for
        line/bbox/ellipsoid.
        """
        parts = [
            self._extract_one_position(df, col_spec)
            for _label, col_spec in self._coordinate_columns
        ]
        return np.column_stack(parts).astype(np.float32)

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
        coords = self._extract_coordinates(df)
        properties, prop_arrays = self._resolve_properties(df)
        rel_names, rel_data = self._extract_relationships(df)
        ids = self._extract_ids(df)

        writer = _PrecomputedAnnotationWriter(
            annotation_type=self.annotation_type,
            coordinate_space=self.coordinate_space,
            data_resolution=self.data_resolution,
            relationships=rel_names,
            properties=properties,
            spatial_hierarchy=self.spatial_hierarchy,
            write_sharded=self.write_sharded,
        )

        writer.set_coordinates(coords)
        writer.set_ids(ids)

        for prop_name, arr in prop_arrays.items():
            writer.set_property(prop_name, arr)

        for rel_name, data in rel_data.items():
            writer.set_relationship(rel_name, data)

        writer.write(path)


# ── Public geometry-specific writers ────────────────────────────────


class PointAnnotationWriter(_AnnotationWriter):
    """Write a DataFrame of point annotations to precomputed format.

    Parameters
    ----------
    point_column : str or list of str
        Column(s) for position coordinates. Accepts:

        - A single column name containing ``[x, y, z]`` arrays
          (e.g., ``"pt_position"``).
        - A column name prefix that expands to ``{prefix}_x``,
          ``{prefix}_y``, ``{prefix}_z``
          (e.g., ``"ctr_pt_position"``).
        - An explicit list of three column names for x, y, z
          (e.g., ``["pt_x", "pt_y", "pt_z"]``).
    segmentation_source : CAVEclient, CloudVolume, or str, optional
        Source to derive coordinate space from. Accepts a CAVEclient
        (uses its segmentation source), a CloudVolume instance, or a
        segmentation source URL. Mutually exclusive with
        ``coordinate_space`` and ``resolution``.
    coordinate_space : CoordinateSpace, optional
        Neuroglancer coordinate space defining the target resolution and axes.
        Mutually exclusive with ``segmentation_source`` and ``resolution``.
    resolution : sequence of float, optional
        Resolution of the target neuroglancer coordinate space per axis
        (e.g., ``[8, 8, 40]``). Units default to nm, axis names to x/y/z.
        Not to be confused with ``data_resolution``, which specifies the
        resolution of the input data. Mutually exclusive with
        ``segmentation_source`` and ``coordinate_space``.
    data_resolution : sequence of float, optional
        Resolution of the input coordinate data (e.g., ``[4, 4, 40]``).
        If provided, coordinates are rescaled from ``data_resolution``
        into the ``coordinate_space`` resolution before writing.
    property_columns : str or list of str, optional
        Column name(s) to include as annotation properties.
    relationship_columns : str or list of str, optional
        Column name(s) to include as annotation relationships.
    id_column : str, optional
        Column for annotation IDs. If None, uses DataFrame index.
    chunk_size : float or array-like, optional
        Spatial index chunk size for the finest level of the spatial hierarchy.
        If set, uses uniform-grid spatial indexing.
    limit : int
        Target max annotations per spatial chunk.
    write_sharded : bool
        Use sharded writes (default True).

    Notes
    -----
    This module requires ``tensorstore``. Install it with::

        pip install 'nglui[precomputed]'

    We leverage tensorstore for writing to both local and cloud storage. Please see the
    `tensorstore documentation <https://google.github.io/tensorstore/kvstore/gcs/index.html#gcs-authentication>`_
    for information on setting up authentication for writing to cloud buckets.

    More information about the Neuroglancer precomputed annotation format is available
    `here <https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/annotations.md>`_.

    The implementation of the spatial hierarchy used in this implementation currently
    assumes an approximately uniform distribution of annotations in space when deciding
    on the structure of the hierarchy. Data-adaptive hierarchy would be a welcome
    contribution, though it may be slower to build.
    """

    def __init__(
        self,
        segmentation_source=None,
        coordinate_space: Optional[CoordinateSpace] = None,
        resolution: Optional[Sequence[float]] = None,
        data_resolution: Optional[Sequence[float]] = None,
        point_column: Optional[Union[str, list[str]]] = None,
        property_columns: Optional[Union[str, list[str]]] = None,
        relationship_columns: Optional[Union[str, list[str]]] = None,
        id_column: Optional[str] = None,
        chunk_size: Optional[Union[float, Sequence[float]]] = None,
        limit: int = 10_000,
    ):
        if point_column is None:
            raise ValueError("point_column is required.")
        super().__init__(
            annotation_type="point",
            coordinate_columns=[("point", point_column)],
            segmentation_source=segmentation_source,
            coordinate_space=coordinate_space,
            resolution=resolution,
            data_resolution=data_resolution,
            property_columns=property_columns,
            relationship_columns=relationship_columns,
            id_column=id_column,
            chunk_size=chunk_size,
            limit=limit,
        )


class LineAnnotationWriter(_AnnotationWriter):
    """Write a DataFrame of line segment annotations to precomputed format.

    Parameters
    ----------
    point_a_column : str or list of str
        Column(s) for position coordinates. Accepts:

        - A single column name containing ``[x, y, z]`` arrays
          (e.g., ``"pt_position"``).
        - A column name prefix that expands to ``{prefix}_x``,
          ``{prefix}_y``, ``{prefix}_z``
          (e.g., ``"ctr_pt_position"``).
        - An explicit list of three column names for x, y, z
          (e.g., ``["pt_x", "pt_y", "pt_z"]``).
    point_b_column : str or list of str
        Column(s) for position coordinates. Accepts:

        - A single column name containing ``[x, y, z]`` arrays
          (e.g., ``"pt_position"``).
        - A column name prefix that expands to ``{prefix}_x``,
          ``{prefix}_y``, ``{prefix}_z``
          (e.g., ``"ctr_pt_position"``).
        - An explicit list of three column names for x, y, z
          (e.g., ``["pt_x", "pt_y", "pt_z"]``).
    segmentation_source : CAVEclient, CloudVolume, or str, optional
        Source to derive coordinate space from. Accepts a CAVEclient
        (uses its segmentation source), a CloudVolume instance, or a
        segmentation source URL. Mutually exclusive with
        ``coordinate_space`` and ``resolution``.
    coordinate_space : CoordinateSpace, optional
        Neuroglancer coordinate space defining the target resolution and axes.
        Mutually exclusive with ``segmentation_source`` and ``resolution``.
    resolution : sequence of float, optional
        Resolution of the target neuroglancer coordinate space per axis
        (e.g., ``[8, 8, 40]``). Units default to nm, axis names to x/y/z.
        Not to be confused with ``data_resolution``, which specifies the
        resolution of the input data. Mutually exclusive with
        ``segmentation_source`` and ``coordinate_space``.
    data_resolution : sequence of float, optional
        Resolution of the input coordinate data (e.g., ``[4, 4, 40]``).
        If provided, coordinates are rescaled from ``data_resolution``
        into the ``coordinate_space`` resolution before writing.
    property_columns : str or list of str, optional
        Column name(s) to include as annotation properties.
    relationship_columns : str or list of str, optional
        Column name(s) to include as annotation relationships.
    id_column : str, optional
        Column for annotation IDs. If None, uses DataFrame index.
    chunk_size : float or array-like, optional
        Spatial index chunk size for the finest level of the spatial hierarchy.
        If set, uses uniform-grid spatial indexing.
    limit : int
        Target max annotations per spatial chunk.
    write_sharded : bool
        Use sharded writes (default True).

    Notes
    -----
    This module requires ``tensorstore``. Install it with::

        pip install 'nglui[precomputed]'

    We leverage tensorstore for writing to both local and cloud storage. Please see the
    `tensorstore documentation <https://google.github.io/tensorstore/kvstore/gcs/index.html#gcs-authentication>`_
    for information on setting up authentication for writing to cloud buckets.

    More information about the Neuroglancer precomputed annotation format is available
    `here <https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/annotations.md>`_.

    The implementation of the spatial hierarchy used in this implementation currently
    assumes an approximately uniform distribution of annotations in space when deciding
    on the structure of the hierarchy. Data-adaptive hierarchy would be a welcome
    contribution, though it may be slower to build.
    """

    def __init__(
        self,
        segmentation_source=None,
        coordinate_space: Optional[CoordinateSpace] = None,
        resolution: Optional[Sequence[float]] = None,
        data_resolution: Optional[Sequence[float]] = None,
        point_a_column: Optional[Union[str, list[str]]] = None,
        point_b_column: Optional[Union[str, list[str]]] = None,
        property_columns: Optional[list[str]] = None,
        relationship_columns: Optional[Union[str, list[str]]] = None,
        id_column: Optional[str] = None,
        chunk_size: Optional[Union[float, Sequence[float]]] = None,
        limit: int = 10_000,
    ):
        if point_a_column is None or point_b_column is None:
            raise ValueError("point_a_column and point_b_column are required.")
        super().__init__(
            annotation_type="line",
            coordinate_columns=[
                ("point_a", point_a_column),
                ("point_b", point_b_column),
            ],
            segmentation_source=segmentation_source,
            coordinate_space=coordinate_space,
            resolution=resolution,
            data_resolution=data_resolution,
            property_columns=property_columns,
            relationship_columns=relationship_columns,
            id_column=id_column,
            chunk_size=chunk_size,
            limit=limit,
        )


class BoundingBoxAnnotationWriter(_AnnotationWriter):
    """Write a DataFrame of axis-aligned bounding box annotations to precomputed format.

    Parameters
    ----------
    point_a_column : str or list of str
        Column(s) for position coordinates. Accepts:

        - A single column name containing ``[x, y, z]`` arrays
          (e.g., ``"pt_position"``).
        - A column name prefix that expands to ``{prefix}_x``,
          ``{prefix}_y``, ``{prefix}_z``
          (e.g., ``"ctr_pt_position"``).
        - An explicit list of three column names for x, y, z
          (e.g., ``["pt_x", "pt_y", "pt_z"]``).
    point_b_column : str or list of str
        Column(s) for position coordinates. Accepts:

        - A single column name containing ``[x, y, z]`` arrays
          (e.g., ``"pt_position"``).
        - A column name prefix that expands to ``{prefix}_x``,
          ``{prefix}_y``, ``{prefix}_z``
          (e.g., ``"ctr_pt_position"``).
        - An explicit list of three column names for x, y, z
          (e.g, ``["pt_x", "pt_y", "pt_z"]``).
    segmentation_source : CAVEclient, CloudVolume, or str, optional
        Source to derive coordinate space from. Accepts a CAVEclient
        (uses its segmentation source), a CloudVolume instance, or a
        segmentation source URL. Mutually exclusive with
        ``coordinate_space`` and ``resolution``.
    coordinate_space : CoordinateSpace, optional
        Neuroglancer coordinate space defining the target resolution and axes.
        Mutually exclusive with ``segmentation_source`` and ``resolution``.
    resolution : sequence of float, optional
        Resolution of the target neuroglancer coordinate space per axis
        (e.g., ``[8, 8, 40]``). Units default to nm, axis names to x/y/z.
        Not to be confused with ``data_resolution``, which specifies the
        resolution of the input data. Mutually exclusive with
        ``segmentation_source`` and ``coordinate_space``.
    data_resolution : sequence of float, optional
        Resolution of the input coordinate data (e.g., ``[4, 4, 40]``).
        If provided, coordinates are rescaled from ``data_resolution``
        into the ``coordinate_space`` resolution before writing.
    property_columns : str or list of str, optional
        Column name(s) to include as annotation properties.
    relationship_columns : str or list of str, optional
        Column name(s) to include as annotation relationships.
    id_column : str, optional
        Column for annotation IDs. If None, uses DataFrame index.
    chunk_size : float or array-like, optional
        Spatial index chunk size for the finest level of the spatial hierarchy.
        If set, uses uniform-grid spatial indexing.
    limit : int
        Target max annotations per spatial chunk.
    write_sharded : bool
        Use sharded writes (default True).

    Notes
    -----
    This module requires ``tensorstore``. Install it with::

        pip install 'nglui[precomputed]'

    We leverage tensorstore for writing to both local and cloud storage. Please see the
    `tensorstore documentation <https://google.github.io/tensorstore/kvstore/gcs/index.html#gcs-authentication>`_
    for information on setting up authentication for writing to cloud buckets.

    More information about the Neuroglancer precomputed annotation format is available
    `here <https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/annotations.md>`_.

    The implementation of the spatial hierarchy used in this implementation currently
    assumes an approximately uniform distribution of annotations in space when deciding
    on the structure of the hierarchy. Data-adaptive hierarchy would be a welcome
    contribution, though it may be slower to build.
    """

    def __init__(
        self,
        segmentation_source=None,
        coordinate_space: Optional[CoordinateSpace] = None,
        resolution: Optional[Sequence[float]] = None,
        data_resolution: Optional[Sequence[float]] = None,
        point_a_column: Optional[Union[str, list[str]]] = None,
        point_b_column: Optional[Union[str, list[str]]] = None,
        property_columns: Optional[list[str]] = None,
        relationship_columns: Optional[Union[str, list[str]]] = None,
        id_column: Optional[str] = None,
        chunk_size: Optional[Union[float, Sequence[float]]] = None,
        limit: int = 10_000,
    ):
        if point_a_column is None or point_b_column is None:
            raise ValueError("point_a_column and point_b_column are required.")
        super().__init__(
            annotation_type="axis_aligned_bounding_box",
            coordinate_columns=[
                ("point_a", point_a_column),
                ("point_b", point_b_column),
            ],
            segmentation_source=segmentation_source,
            coordinate_space=coordinate_space,
            resolution=resolution,
            data_resolution=data_resolution,
            property_columns=property_columns,
            relationship_columns=relationship_columns,
            id_column=id_column,
            chunk_size=chunk_size,
            limit=limit,
        )


class EllipsoidAnnotationWriter(_AnnotationWriter):
    """Write a DataFrame of ellipsoid annotations to precomputed format.

    Parameters
    ----------
    center_column : str or list of str
        Column(s) for position coordinates. Accepts:

        - A single column name containing ``[x, y, z]`` arrays
          (e.g., ``"pt_position"``).
        - A column name prefix that expands to ``{prefix}_x``,
          ``{prefix}_y``, ``{prefix}_z``
          (e.g., ``"ctr_pt_position"``).
        - An explicit list of three column names for x, y, z
          (e.g., ``["pt_x", "pt_y", "pt_z"]``).
    radii_column : str or list of str
        Column(s) for position coordinates. Accepts:

        - A single column name containing ``[x, y, z]`` arrays
          (e.g., ``"pt_position"``).
        - A column name prefix that expands to ``{prefix}_x``,
          ``{prefix}_y``, ``{prefix}_z``
          (e.g., ``"ctr_pt_position"``).
        - An explicit list of three column names for x, y, z
          (e.g., ``["pt_x", "pt_y", "pt_z"]``).
    segmentation_source : CAVEclient, CloudVolume, or str, optional
        Source to derive coordinate space from. Accepts a CAVEclient
        (uses its segmentation source), a CloudVolume instance, or a
        segmentation source URL. Mutually exclusive with
        ``coordinate_space`` and ``resolution``.
    coordinate_space : CoordinateSpace, optional
        Neuroglancer coordinate space defining the target resolution and axes.
        Mutually exclusive with ``segmentation_source`` and ``resolution``.
    resolution : sequence of float, optional
        Resolution of the target neuroglancer coordinate space per axis
        (e.g., ``[8, 8, 40]``). Units default to nm, axis names to x/y/z.
        Not to be confused with ``data_resolution``, which specifies the
        resolution of the input data. Mutually exclusive with
        ``segmentation_source`` and ``coordinate_space``.
    data_resolution : sequence of float, optional
        Resolution of the input coordinate data (e.g., ``[4, 4, 40]``).
        If provided, coordinates are rescaled from ``data_resolution``
        into the ``coordinate_space`` resolution before writing.
    property_columns : str or list of str, optional
        Column name(s) to include as annotation properties.
    relationship_columns : str or list of str, optional
        Column name(s) to include as annotation relationships.
    id_column : str, optional
        Column for annotation IDs. If None, uses DataFrame index.
    chunk_size : float or array-like, optional
        Spatial index chunk size for the finest level of the spatial hierarchy.
        If set, uses uniform-grid spatial indexing.
    limit : int
        Target max annotations per spatial chunk.
    write_sharded : bool
        Use sharded writes (default True).

    Notes
    -----
    This module requires ``tensorstore``. Install it with::

        pip install 'nglui[precomputed]'

    We leverage tensorstore for writing to both local and cloud storage. Please see the
    `tensorstore documentation <https://google.github.io/tensorstore/kvstore/gcs/index.html#gcs-authentication>`_
    for information on setting up authentication for writing to cloud buckets.

    More information about the Neuroglancer precomputed annotation format is available
    `here <https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/annotations.md>`_.

    The implementation of the spatial hierarchy used in this implementation currently
    assumes an approximately uniform distribution of annotations in space when deciding
    on the structure of the hierarchy. Data-adaptive hierarchy would be a welcome
    contribution, though it may be slower to build.
    """

    def __init__(
        self,
        segmentation_source=None,
        coordinate_space: Optional[CoordinateSpace] = None,
        resolution: Optional[Sequence[float]] = None,
        data_resolution: Optional[Sequence[float]] = None,
        center_column: Optional[Union[str, list[str]]] = None,
        radii_column: Optional[Union[str, list[str]]] = None,
        property_columns: Optional[Union[str, list[str]]] = None,
        relationship_columns: Optional[Union[str, list[str]]] = None,
        id_column: Optional[str] = None,
        chunk_size: Optional[Union[float, Sequence[float]]] = None,
        limit: int = 10_000,
    ):
        if center_column is None or radii_column is None:
            raise ValueError("center_column and radii_column are required.")
        super().__init__(
            annotation_type="ellipsoid",
            coordinate_columns=[("center", center_column), ("radii", radii_column)],
            segmentation_source=segmentation_source,
            coordinate_space=coordinate_space,
            resolution=resolution,
            data_resolution=data_resolution,
            property_columns=property_columns,
            relationship_columns=relationship_columns,
            id_column=id_column,
            chunk_size=chunk_size,
            limit=limit,
        )
