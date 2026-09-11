"""Golden-format tests for serialized annotation tags.

These pin the exact JSON that reaches Neuroglancer. Tag encoding is a wire format:
the viewer reads `annotationProperties` positionally, so the contents *and ordering*
of these structures are contractual, not incidental.

The legacy (spelunker) expectations here were captured from the implementation before
capability-based format selection was introduced, and must not drift.
"""

import pandas as pd
import pytest

from nglui.statebuilder import ViewerState


@pytest.fixture
def tagged_df():
    """Two points, one string tag column and one boolean tag column."""
    return pd.DataFrame(
        {
            "x": [1, 2],
            "y": [3, 4],
            "z": [5, 6],
            "ct": ["a", "b"],
            "flag": [True, False],
        }
    )


def _anno_layer(df, **kwargs):
    vs = ViewerState(dimensions=[1, 1, 1])
    vs.add_points(
        df,
        point_column=["x", "y", "z"],
        tag_column="ct",
        tag_bools=["flag"],
        linked_segmentation=None,
        **kwargs,
    )
    return vs.to_dict()["layers"][0]


class TestLegacyTagWireFormat:
    def test_annotation_properties(self, tagged_df):
        """Legacy tags are uint8 properties carrying a non-standard `tag` key."""
        layer = _anno_layer(tagged_df)
        assert layer["annotationProperties"] == [
            {"id": "tag0", "type": "uint8", "tag": "a"},
            {"id": "tag1", "type": "uint8", "tag": "b"},
            {"id": "tag2", "type": "uint8", "tag": "flag"},
        ]

    def test_tool_bindings(self, tagged_df):
        """Bindings point at the tagTool_* types nglui registers with neuroglancer."""
        layer = _anno_layer(tagged_df)
        assert layer["toolBindings"] == {
            "Q": "tagTool_tag0",
            "W": "tagTool_tag1",
            "E": "tagTool_tag2",
        }

    def test_props_are_positional_bit_vectors(self, tagged_df):
        """Each annotation carries one 0/1 per property, in property order."""
        layer = _anno_layer(tagged_df)
        props = [a["props"] for a in layer["annotations"]]
        # row 0: ct == "a" and flag is True -> tag0 and tag2
        # row 1: ct == "b" and flag is False -> tag1 only
        assert props == [[1, 0, 1], [0, 1, 0]]

    def test_props_length_matches_property_count(self, tagged_df):
        """The viewer requires exactly one props entry per declared property."""
        layer = _anno_layer(tagged_df)
        n_props = len(layer["annotationProperties"])
        assert all(len(a["props"]) == n_props for a in layer["annotations"])

    def test_tag_ordering_is_explicit_then_alphabetical(self):
        """Explicit tags keep their order; discovered tags append alphabetically."""
        df = pd.DataFrame(
            {"x": [1], "y": [2], "z": [3], "ct": ["zebra"], "apple": [True]}
        )
        vs = ViewerState(dimensions=[1, 1, 1])
        vs.add_points(
            df,
            point_column=["x", "y", "z"],
            tag_column="ct",
            tag_bools=["apple"],
            tags=["explicit"],
            linked_segmentation=None,
        )
        layer = vs.to_dict()["layers"][0]
        assert [p["tag"] for p in layer["annotationProperties"]] == [
            "explicit",
            "apple",
            "zebra",
        ]

    def test_local_source_and_no_tags(self):
        """A layer with no tags declares no properties and no bindings."""
        df = pd.DataFrame({"x": [1], "y": [2], "z": [3]})
        vs = ViewerState(dimensions=[1, 1, 1])
        vs.add_points(df, point_column=["x", "y", "z"], linked_segmentation=None)
        layer = vs.to_dict()["layers"][0]
        assert layer["annotationProperties"] == []
        assert layer.get("toolBindings", {}) == {}
