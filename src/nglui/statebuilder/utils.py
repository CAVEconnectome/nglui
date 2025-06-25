import numbers
import re
from collections.abc import Iterable, Mapping
from typing import Union
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import webcolors


class NamedList(list):
    """A list that can be indexed by name."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._name_map = {str(item.name): item for item in self}

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._name_map[key]
        return super().__getitem__(key)

    def append(self, item):
        super().append(item)
        self._name_map[str(item.name)] = item

    def extend(self, items):
        super().extend(items)
        for item in items:
            self._name_map[str(item.name)] = item


def split_point_columns(col_name: str, columns: list[str]) -> Union[str, list[str]]:
    if col_name in columns:
        return col_name
    suffixes = ["x", "y", "z"]
    resolved_columns = [f"{col_name}_{suffix}" for suffix in suffixes]

    if all(col in columns for col in resolved_columns):
        return resolved_columns
    raise ValueError(
        f"Column '{col_name}' not found, and '{col_name}_x', '{col_name}_y', '{col_name}_z' are not all present in the columns."
    )


def strip_numpy_types(obj):
    """
    Recursively convert numpy types in scalars, list-like, or dictionary values
    into pure Python types.

    Parameters
    ----------
    obj : any
        The object to process (scalar, list-like, or dictionary).

    Returns
    -------
    any
        The object with all numpy types converted to pure Python types.
    """
    if isinstance(obj, (np.generic, np.ndarray)):
        # Convert numpy scalars or arrays to Python types or lists
        if np.isscalar(obj):
            return obj.item()
        else:
            return [strip_numpy_types(x) for x in obj]
    elif isinstance(obj, pd.Series):
        # Convert pandas Series to a list of Python types
        return [strip_numpy_types(x) for x in obj.values]
    elif isinstance(obj, Mapping):
        # Recursively process dictionary values
        return {strip_numpy_types(k): strip_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
        # Recursively process list-like objects
        return [strip_numpy_types(x) for x in obj]
    else:
        # Return the object as-is if it's already a pure Python type
        return obj


def list_of_lists(obj):
    "Strip numpy types and return as a list of lists. Should not be used for dicts."
    if pd.isnull(obj):
        return None
    obj = strip_numpy_types(obj)
    if isinstance(obj, list):
        return [obj]
    else:
        return [[obj]]


def list_of_strings(obj):
    if isinstance(obj, str):
        return [obj]
    elif isinstance(obj, Iterable):
        return [str(x) for x in obj if not pd.isna(x)]
    else:
        return []


def none_or_array(obj):
    if obj:
        return np.array(obj)
    else:
        return None


def is_list_like(obj):
    """
    Check if an object is list-like (iterable but not a mapping or string).

    Parameters
    ----------
    obj : any
        Object to check

    Returns
    -------
    bool
        True if the object is list-like, False otherwise
    """
    return (
        isinstance(obj, Iterable)
        and not isinstance(obj, Mapping)
        and not isinstance(obj, str)
    )


def is_dict_like(obj):
    """
    Check if an object is dictionary-like (mapping).

    Parameters
    ----------
    obj : any
        Object to check

    Returns
    -------
    bool
        True if the object is dictionary-like, False otherwise
    """
    return isinstance(obj, Mapping) and not isinstance(obj, str)


def omit_nones(seg_list):
    if seg_list is None or np.all(pd.isna(seg_list)):
        return []
    seg_list = np.atleast_1d(seg_list)
    seg_list = list(filter(lambda x: x is not None, seg_list))
    if len(seg_list) == 0:
        return []
    else:
        return seg_list


def parse_color(clr):
    if clr is None:
        return None

    if isinstance(clr, numbers.Number):
        clr = (clr, clr, clr)

    if isinstance(clr, str):
        hex_match = r"\#[0123456789abcdef]{6}"
        if re.match(hex_match, clr.lower()):
            return clr
        else:
            return webcolors.name_to_hex(clr)
    else:
        return webcolors.rgb_to_hex([int(255 * x) for x in clr])


def parse_graphene_header(source):
    qry = urlparse(source)
    if qry.scheme == "graphene":
        return _parse_to_mainline(qry)
    else:
        return source


def parse_graphene_image_url(source):
    qry = urlparse(source)
    if qry.scheme == "graphene":
        return _parse_to_mainline_imagery(qry)
    else:
        return source


def _parse_to_mainline(qry):
    if "https" in qry.netloc and "middleauth":
        return f"{qry.scheme}://middleauth+https:{qry.path}"
    else:
        return f"{qry.scheme}://middleauth+http:{qry.path}"


def _parse_to_mainline_imagery(qry):
    if qry.scheme == "graphene":
        if "https" in qry.netloc:
            return f"precomputed://middleauth+https:{qry.path}"
        else:
            return f"precomputed://middleauth+http:{qry.path}"
    else:
        return qry.geturl()
