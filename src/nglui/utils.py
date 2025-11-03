"""Utility functions for nglui that are shared across modules."""

from typing import Union

import pandas as pd


def convert_arrow_to_numpy(
    df: pd.DataFrame, columns: Union[str, list[str], None] = None
) -> pd.DataFrame:
    """Ensure DataFrame is compatible with segment properties operations.

    Converts PyArrow-backed columns to pandas/numpy dtypes.
    Uses explicit type mapping for full control over conversions.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to convert
    columns : str, list[str], or None, optional
        Specific column(s) to convert. If None, converts all PyArrow-backed columns.
        Can be a single column name (str) or a list of column names.
        Lists can be nested (e.g., [['x', 'y', 'z'], 'id']) and will be flattened.
        None values in the list are automatically filtered out.

    Returns
    -------
    pd.DataFrame
        DataFrame with specified columns converted from PyArrow to numpy/pandas dtypes

    Note
    ----
    pd.StringDtype("pyarrow") and pd.ArrowDtype(pa.string()) both
    display as "string[pyarrow]" but are different types.

    Examples
    --------
    >>> # Convert all PyArrow columns
    >>> df_converted = convert_arrow_to_numpy(df)
    >>>
    >>> # Convert only specific columns
    >>> df_converted = convert_arrow_to_numpy(df, columns=['id', 'name'])
    >>>
    >>> # Convert a single column
    >>> df_converted = convert_arrow_to_numpy(df, columns='id')
    >>>
    >>> # Works with nested lists (useful for point columns)
    >>> df_converted = convert_arrow_to_numpy(df, columns=[['x', 'y', 'z'], 'segment_id'])
    >>>
    >>> # None values are automatically filtered out (useful for optional columns)
    >>> df_converted = convert_arrow_to_numpy(
    ...     df, columns=[['x', 'y', 'z'], segment_col, description_col]
    ... )  # segment_col and description_col can be None
    """
    # Determine which columns to check/convert
    if columns is None:
        # Convert all columns (original behavior)
        cols_to_check = df.columns
    else:
        # Normalize columns to a flat list, filtering out None values
        if isinstance(columns, str):
            cols_to_check = [columns]
        else:
            # Flatten nested lists and filter out None values
            cols_to_check = []
            for col in columns:
                if col is None:
                    continue  # Skip None values
                elif isinstance(col, list):
                    cols_to_check.extend(col)
                else:
                    cols_to_check.append(col)

        # Filter to only columns that exist in the DataFrame
        cols_to_check = [col for col in cols_to_check if col in df.columns]

    # If no columns to check, return original DataFrame
    if len(cols_to_check) == 0:
        return df

    # Check if any of the specified columns have PyArrow backing
    has_arrow = any(isinstance(df[col].dtype, pd.ArrowDtype) for col in cols_to_check)
    has_pyarrow_string = any(
        isinstance(df[col].dtype, pd.StringDtype) and df[col].dtype.storage == "pyarrow"
        for col in cols_to_check
    )

    if not (has_arrow or has_pyarrow_string):
        return df

    df_converted = df.copy()

    # Manually convert each specified PyArrow-backed column
    for col in cols_to_check:
        # Handle pd.StringDtype with pyarrow storage - convert to object
        if isinstance(df_converted[col].dtype, pd.StringDtype):
            if df_converted[col].dtype.storage == "pyarrow":
                df_converted[col] = df_converted[col].astype(object)
                continue

        # Handle pd.ArrowDtype columns
        if isinstance(df_converted[col].dtype, pd.ArrowDtype):
            if hasattr(df_converted[col].dtype, "pyarrow_dtype"):
                import pyarrow as pa

                pa_type = df_converted[col].dtype.pyarrow_dtype

                # Handle timestamp columns
                if pa.types.is_timestamp(pa_type):
                    if pa_type.tz is not None:
                        # Convert timezone-aware to UTC, remove timezone, then convert to numpy
                        df_converted[col] = (
                            df_converted[col]
                            .dt.tz_convert("UTC")
                            .dt.tz_localize(None)
                            .astype("datetime64[ns]")
                        )
                    else:
                        # Convert timezone-naive timestamp to numpy
                        df_converted[col] = df_converted[col].astype("datetime64[ns]")

                # Handle string columns - convert to object
                elif pa.types.is_string(pa_type) or pa.types.is_large_string(pa_type):
                    df_converted[col] = df_converted[col].astype(object)

                # Handle integer types
                elif pa.types.is_integer(pa_type):
                    # Use pandas nullable integer types
                    if pa.types.is_int64(pa_type):
                        df_converted[col] = df_converted[col].astype("Int64")
                    elif pa.types.is_int32(pa_type):
                        df_converted[col] = df_converted[col].astype("Int32")
                    elif pa.types.is_int16(pa_type):
                        df_converted[col] = df_converted[col].astype("Int16")
                    elif pa.types.is_int8(pa_type):
                        df_converted[col] = df_converted[col].astype("Int8")
                    elif pa.types.is_uint64(pa_type):
                        df_converted[col] = df_converted[col].astype("UInt64")
                    elif pa.types.is_uint32(pa_type):
                        df_converted[col] = df_converted[col].astype("UInt32")
                    elif pa.types.is_uint16(pa_type):
                        df_converted[col] = df_converted[col].astype("UInt16")
                    elif pa.types.is_uint8(pa_type):
                        df_converted[col] = df_converted[col].astype("UInt8")
                    else:
                        df_converted[col] = df_converted[col].astype("Int64")

                # Handle float types
                elif pa.types.is_floating(pa_type):
                    if pa.types.is_float64(pa_type):
                        df_converted[col] = df_converted[col].astype("Float64")
                    elif pa.types.is_float32(pa_type):
                        df_converted[col] = df_converted[col].astype("Float32")
                    else:
                        df_converted[col] = df_converted[col].astype("Float64")

                # Handle boolean types
                elif pa.types.is_boolean(pa_type):
                    df_converted[col] = df_converted[col].astype("boolean")

                # Fallback for any other PyArrow types
                else:
                    df_converted[col] = df_converted[col].astype(object)

    return df_converted
