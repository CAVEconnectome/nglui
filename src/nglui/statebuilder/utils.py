import re
from collections.abc import Iterable

import numpy as np

FALLBACK_SEUNGLAB_NGL_URL = "https://neuroglancer.neuvue.io"
FALLBACK_MAINLINE_NGL_URL = "https://ngl.cave-explorer.org"

SPLIT_SUFFIXES = ['x', 'y', 'z']

def bucket_of_values(col, data, item_is_array=False, array_width=3):
    """
    Use to get a flat array of items when you don't know if it's already a collection or a collection of iterables.
    Parameters:
        dataseries: Pandas dataseries with either all items or all a collection of items.
                    If the item is expected to be an nd-array, use item_shape to define what an element is.
    """

    if len(data) == 0:
        return []

    dataseries = data[col]
    dataseries = dataseries[~dataseries.isnull()]

    if item_is_array:
        # If already an m x n array, just vstack. Else, need to stack every element first.
        if type(dataseries.iloc[0]) is np.ndarray:
            if len(data) > 1:
                return np.vstack(dataseries.values)
            else:
                return dataseries.values[0].reshape(1, -1)
        elif len(data) > 1:
            return np.vstack(dataseries.map(np.vstack)).reshape(-1, array_width)
        else:
            return np.vstack(dataseries).reshape(1, -1)
    elif isinstance(dataseries.iloc[0], Iterable):
        return np.concatenate(dataseries.values)
    else:
        return dataseries.values


def is_split_position(pt_col, df, suffixes=SPLIT_SUFFIXES):
    if pt_col in df.columns:
        return True
    prefix_found = []
    for suf in suffixes:
        prefix_found.append( np.any([re.search(f"^{pt_col}_{suf}", col) is not None for col in df.columns]) )
    if np.all(prefix_found):
        return True
    else:
        raise ValueError(f'Point column "{pt_col}" not found directly or as split position')

def is_split_split_position(pt_col, df, suffixes=SPLIT_SUFFIXES):
    is_split_split = []
    for suf in suffixes:
        split_name = pt_col.split('_')
        expected_name = '_'.join(split_name[:-1]) + f'_{suf}_{split_name[-1]}'
        is_split_split.append(expected_name in df.columns)
    return np.all(is_split_split)

def assemble_split_points(pt_col, df, suffixes=SPLIT_SUFFIXES):
    cols = split_position_columns(pt_col, suffixes)
    return np.vstack(df[cols].values)


def split_position_columns(pt_col, suffixes=SPLIT_SUFFIXES):
    return [f"{pt_col}_{suf}" for suf in suffixes]



def check_target_site(ngl_url, client):
    """
    Check neuroglancer info to determine which kind of site a neuroglancer URL is.
    """
    ngl_info = client.state.get_neuroglancer_info(ngl_url)
    if len(ngl_info)==0:
        return 'seunglab'
    else:
        return "cave-explorer"
