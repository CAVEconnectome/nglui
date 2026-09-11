"""Reading tags back from all three Neuroglancer state dialects.

Tags have been encoded three different ways across Neuroglancer forks and versions,
and the parser has to read states produced by any of them.
"""

import pandas as pd
import pytest

from nglui import parser
from nglui.parser.base import annotation_property_dictionary
from nglui.statebuilder import ViewerState

LEGACY_ANNOTATION_TAGS_STATE = {
    "navigation": {"pose": {"position": {"voxelSize": [4, 4, 40]}}},
    "layers": [
        {
            "name": "annos",
            "type": "annotation",
            "annotationTags": [{"id": 0, "label": "axon"}, {"id": 1, "label": "soma"}],
            "annotations": [
                {"type": "point", "point": [1, 2, 3], "tagIds": [0]},
                {"type": "point", "point": [4, 5, 6], "tagIds": [0, 1]},
            ],
        }
    ],
}


def _build_state(capabilities):
    df = pd.DataFrame({"x": [1, 4], "y": [2, 5], "z": [3, 6], "ct": ["axon", "soma"]})
    vs = ViewerState(dimensions=[1, 1, 1], capabilities=capabilities)
    vs.add_points(
        df,
        name="annos",
        point_column=["x", "y", "z"],
        tag_column="ct",
        linked_segmentation=None,
    )
    return vs.to_dict()


class TestTagDictionary:
    def test_reads_annotation_tags_dialect(self):
        assert parser.tag_dictionary(LEGACY_ANNOTATION_TAGS_STATE, "annos") == {
            0: "axon",
            1: "soma",
        }

    def test_reads_spelunker_tag_property_dialect(self):
        state = _build_state("legacy")
        assert parser.tag_dictionary(state, "annos") == {0: "axon", 1: "soma"}

    def test_reads_bool_property_dialect(self):
        state = _build_state("main")
        assert parser.tag_dictionary(state, "annos") == {0: "axon", 1: "soma"}

    def test_bool_property_label_prefers_description(self):
        """A sanitized id keeps its original label in `description`."""
        df = pd.DataFrame({"x": [1], "y": [2], "z": [3], "Cell Body": [True]})
        vs = ViewerState(dimensions=[1, 1, 1], capabilities="main")
        with pytest.warns(UserWarning):
            vs.add_points(
                df,
                name="annos",
                point_column=["x", "y", "z"],
                tag_bools=["Cell Body"],
                linked_segmentation=None,
            )
            state = vs.to_dict()
        assert parser.tag_dictionary(state, "annos") == {0: "Cell Body"}


class TestAnnotationTagRoundTrip:
    @pytest.mark.parametrize("capabilities", ["legacy", "main"])
    def test_tags_survive_a_round_trip(self, capabilities):
        state = _build_state(capabilities)
        _, tag_ids = parser.point_annotations(state, "annos", tags=True)
        labels = parser.tag_dictionary(state, "annos")
        assert [[labels[i] for i in ids] for ids in tag_ids] == [["axon"], ["soma"]]

    def test_legacy_annotation_tags_round_trip(self):
        _, tag_ids = parser.point_annotations(
            LEGACY_ANNOTATION_TAGS_STATE, "annos", tags=True
        )
        assert tag_ids == [[0], [0, 1]]

    @pytest.mark.parametrize("capabilities", ["legacy", "main"])
    def test_expand_tags_dataframe(self, capabilities):
        state = _build_state(capabilities)
        df = parser.annotation_dataframe(state, expand_tags=True)
        assert df["axon"].tolist() == [True, False]
        assert df["soma"].tolist() == [False, True]


class TestNonTagProperties:
    def test_numeric_properties_are_not_read_as_tags(self):
        """A numeric value of 1 is data, not a set tag."""
        state = _build_state("main")
        layer = state["layers"][0]
        layer["annotationProperties"].append({"id": "score", "type": "float32"})
        for anno in layer["annotations"]:
            anno["props"].append(1.0)
        _, tag_ids = parser.point_annotations(state, "annos", tags=True)
        assert tag_ids == [[0], [1]]

    def test_property_dictionary_reports_every_property(self):
        state = _build_state("main")
        state["layers"][0]["annotationProperties"].append(
            {"id": "score", "type": "float32"}
        )
        props = annotation_property_dictionary(state, "annos")
        assert set(props) == {"axon", "soma", "score"}
        assert props["score"]["type"] == "float32"
