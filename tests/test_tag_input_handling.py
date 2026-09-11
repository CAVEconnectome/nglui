"""Tag columns as they actually arrive from real dataframes.

Tag inputs come straight from user data: a cell-type column with gaps, a flag column
read back as floats because it had a null, ids that are numbers. Each of these used to
crash or silently lose data.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from nglui.statebuilder import ViewerState

BASE = {"x": [1, 2, 3], "y": [1, 2, 3], "z": [1, 2, 3]}


def layer(capabilities="main", **kwargs):
    vs = ViewerState(dimensions=[1, 1, 1], capabilities=capabilities)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vs.add_points(point_column=["x", "y", "z"], linked_segmentation=None, **kwargs)
        return vs.to_dict()["layers"][0]


def ids_and_props(lyr):
    return (
        [p["id"] for p in lyr["annotationProperties"]],
        [a["props"] for a in lyr["annotations"]],
    )


class TestMissingValuesInTagColumn:
    """A cell-type column with gaps is ordinary data, and used to raise TypeError.

    The layer's tag vocabulary was built with `sorted()` over the column's unique
    values, which cannot order a float NaN against strings.
    """

    @pytest.mark.parametrize("missing", [np.nan, None, "", "   "])
    def test_missing_means_untagged_not_a_tag(self, missing):
        lyr = layer(
            data=pd.DataFrame({**BASE, "ct": ["a", missing, "b"]}), tag_column="ct"
        )
        ids, props = ids_and_props(lyr)
        assert ids == ["a", "b"]
        assert props[1] == [False, False]

    def test_every_value_missing(self):
        lyr = layer(data=pd.DataFrame({**BASE, "ct": [np.nan] * 3}), tag_column="ct")
        ids, props = ids_and_props(lyr)
        assert ids == []
        assert props == [[], [], []]

    @pytest.mark.parametrize("capabilities", ["main", "legacy"])
    def test_props_stay_full_length(self, capabilities):
        lyr = layer(
            capabilities=capabilities,
            data=pd.DataFrame({**BASE, "ct": ["a", None, "b"]}),
            tag_column="ct",
        )
        ids, props = ids_and_props(lyr)
        assert all(len(p) == len(ids) for p in props)


class TestMissingValuesInTagBools:
    """A flag column with one null comes back as float or nullable dtype.

    Indexing with the raw column then raises IndexError, because it is no longer a
    boolean array.
    """

    @pytest.mark.parametrize(
        "column",
        [
            [True, np.nan, False],
            pd.array([1, 0, None], dtype="Int64"),
            pd.array([True, None, False], dtype="boolean"),
            [1, 0, 0],
        ],
    )
    def test_missing_counts_as_not_tagged(self, column):
        lyr = layer(data=pd.DataFrame({**BASE, "f": column}), tag_bools=["f"])
        ids, props = ids_and_props(lyr)
        assert ids == ["f"]
        assert props == [[True], [False], [False]]

    def test_all_false_still_declares_the_property(self):
        lyr = layer(data=pd.DataFrame({**BASE, "f": [False] * 3}), tag_bools=["f"])
        ids, props = ids_and_props(lyr)
        assert ids == ["f"]
        assert props == [[False]] * 3


class TestNonStringTagValues:
    """Annotations coerce their tags to str, so the vocabulary has to agree.

    It did not, so a numeric tag never matched the annotation carrying it and every
    value read as unset -- with no error, and a property list that looked right.
    """

    @pytest.mark.parametrize("capabilities", ["main", "legacy"])
    def test_numeric_tags_are_actually_set(self, capabilities):
        lyr = layer(
            capabilities=capabilities,
            data=pd.DataFrame({**BASE, "ct": [1, 2, 1]}),
            tag_column="ct",
        )
        ids, props = ids_and_props(lyr)
        assert len(ids) == 2
        assert [sum(bool(v) for v in p) for p in props] == [1, 1, 1]

    def test_mixed_types_do_not_raise_when_ordering(self):
        """`sorted` over ints beside strings needs a key to avoid TypeError."""
        lyr = layer(data=pd.DataFrame({**BASE, "ct": [1, "a", 2]}), tag_column="ct")
        ids, props = ids_and_props(lyr)
        assert len(ids) == 3
        assert [sum(bool(v) for v in p) for p in props] == [1, 1, 1]
