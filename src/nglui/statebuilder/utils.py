from collections.abc import Iterable
import numpy as np


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
        else:
            if len(data) > 1:
                return np.vstack(dataseries.map(np.vstack)).reshape(-1, array_width)
            else:
                return np.vstack(dataseries).reshape(1, -1)
    else:
        if isinstance(dataseries.iloc[0], Iterable):
            return np.concatenate(dataseries.values)
        else:
            return dataseries.values
