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


def _anno_layer(df, capabilities=None, **kwargs):
    vs = ViewerState(dimensions=[1, 1, 1], capabilities=capabilities)
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
    """Pinned explicitly: this describes the encoding, not any live deployment.

    These once relied on the default target resolving to legacy, which made them a
    live probe of Spelunker -- and they broke the day it gained bool properties.
    """

    def test_annotation_properties(self, tagged_df):
        """Legacy tags are uint8 properties carrying a non-standard `tag` key."""
        layer = _anno_layer(tagged_df, capabilities="legacy")
        assert layer["annotationProperties"] == [
            {"id": "tag0", "type": "uint8", "tag": "a"},
            {"id": "tag1", "type": "uint8", "tag": "b"},
            {"id": "tag2", "type": "uint8", "tag": "flag"},
        ]

    def test_tool_bindings(self, tagged_df):
        """Bindings point at the tagTool_* types nglui registers with neuroglancer."""
        layer = _anno_layer(tagged_df, capabilities="legacy")
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
        vs = ViewerState(dimensions=[1, 1, 1], capabilities="legacy")
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


class TestBoolPropertyWireFormat:
    """The encoding used by main Neuroglancer, where a tag is a bool property."""

    def test_annotation_properties(self, tagged_df):
        layer = _anno_layer(tagged_df, capabilities="main")
        assert layer["annotationProperties"] == [
            {"id": "a", "type": "bool"},
            {"id": "b", "type": "bool"},
            {"id": "flag", "type": "bool"},
        ]

    def test_tool_bindings_are_object_form(self, tagged_df):
        layer = _anno_layer(tagged_df, capabilities="main")
        assert layer["toolBindings"] == {
            "Q": {"type": "toggleBoolProperty", "property": "a"},
            "W": {"type": "toggleBoolProperty", "property": "b"},
            "E": {"type": "toggleBoolProperty", "property": "flag"},
        }

    def test_props_are_booleans(self, tagged_df):
        layer = _anno_layer(tagged_df, capabilities="main")
        props = [a["props"] for a in layer["annotations"]]
        assert props == [[True, False, True], [False, True, False]]

    def test_props_length_matches_property_count(self, tagged_df):
        layer = _anno_layer(tagged_df, capabilities="main")
        n_props = len(layer["annotationProperties"])
        assert all(len(a["props"]) == n_props for a in layer["annotations"])

    def test_invalid_label_is_sanitized_and_described(self):
        df = pd.DataFrame({"x": [1], "y": [2], "z": [3], "Cell Body": [True]})
        vs = ViewerState(dimensions=[1, 1, 1], capabilities="main")
        with pytest.warns(UserWarning, match="were renamed"):
            vs.add_points(
                df,
                point_column=["x", "y", "z"],
                tag_bools=["Cell Body"],
                linked_segmentation=None,
            )
            layer = vs.to_dict()["layers"][0]
        assert layer["annotationProperties"] == [
            {"id": "cell_body", "type": "bool", "description": "Cell Body"}
        ]


class TestCapabilityResolution:
    def test_explicit_capabilities_win_over_target(self, tagged_df):
        """A pinned setting must not be second-guessed by a probe."""
        layer = _anno_layer(tagged_df, capabilities="legacy")
        assert layer["annotationProperties"][0]["type"] == "uint8"

    def test_layer_capabilities_override_state(self, tagged_df):
        vs = ViewerState(dimensions=[1, 1, 1], capabilities="legacy")
        vs.add_annotation_layer(name="anno", capabilities="main")
        vs.add_points(
            tagged_df,
            name="anno",
            point_column=["x", "y", "z"],
            tag_column="ct",
            linked_segmentation=None,
        )
        layer = vs.to_dict()["layers"][0]
        assert layer["annotationProperties"][0]["type"] == "bool"

    def test_tagless_state_never_probes(self, mocker):
        """No tags means the encoding cannot differ, so the network is not touched."""
        get = mocker.patch("requests.get")
        df = pd.DataFrame({"x": [1], "y": [2], "z": [3]})
        vs = ViewerState(dimensions=[1, 1, 1])
        vs.add_points(df, point_column=["x", "y", "z"], linked_segmentation=None)
        vs.to_dict()
        assert get.call_count == 0

    def test_pinned_capabilities_never_probe(self, mocker, tagged_df):
        get = mocker.patch("requests.get")
        _anno_layer(tagged_df, capabilities="main")
        assert get.call_count == 0

    def test_cloud_layer_warns_that_tags_are_dropped(self):
        vs = ViewerState(dimensions=[1, 1, 1])
        vs.add_annotation_layer(
            name="anno",
            source="precomputed://gs://example/annotations",
            tags=["axon"],
            linked_segmentation=None,
        )
        with pytest.warns(UserWarning, match="will not be included"):
            vs.to_dict()


class TestLateTargetOverride:
    """`to_url` can name a target after the state was already built and cached."""

    def _properties_from_url(self, url):
        import json
        import urllib.parse

        fragment = urllib.parse.unquote(url.split("#!", 1)[1])
        return json.loads(fragment)["layers"][0]["annotationProperties"]

    @pytest.fixture
    def two_targets(self, mocker):
        """One target supporting bool properties, one not."""
        from nglui.statebuilder import capabilities as caps

        caps.clear_capability_cache()
        modern = 'x={int8:e.INT8,bool:e.UINT8};d="toggleBoolProperty";'
        legacy = 'x={int8:e.INT8};class a{static TOOL_ID="tagTool";}'

        def get(url, **kwargs):
            response = mocker.Mock()
            if url.endswith("/"):
                response.text = '<script src="main.abc.js"></script>'
            else:
                response.text = modern if "modern" in url else legacy
            response.content = response.text.encode()
            response.raise_for_status = mocker.Mock()
            return response

        mocker.patch("requests.get", side_effect=get)
        yield
        caps.clear_capability_cache()

    @pytest.fixture
    def tagged_state(self, tagged_df, two_targets):
        vs = ViewerState(dimensions=[1, 1, 1], target_url="https://legacy.example/")
        vs.add_points(
            tagged_df,
            point_column=["x", "y", "z"],
            tag_column="ct",
            linked_segmentation=None,
        )
        return vs

    def test_target_site_changes_the_encoding(self, tagged_state):
        """Reusing the cached state here would silently ship the wrong encoding."""
        default = self._properties_from_url(tagged_state.to_url())
        google = self._properties_from_url(
            tagged_state.to_url(target_url="https://modern.example/")
        )
        assert default[0]["type"] == "uint8"
        assert google[0]["type"] == "bool"

    def test_original_target_is_restored(self, tagged_state):
        tagged_state.to_url(target_url="https://modern.example/")
        after = self._properties_from_url(tagged_state.to_url())
        assert after[0]["type"] == "uint8"

    def test_pinned_capabilities_ignore_the_target(self, tagged_df):
        vs = ViewerState(dimensions=[1, 1, 1], capabilities="legacy")
        vs.add_points(
            tagged_df,
            point_column=["x", "y", "z"],
            tag_column="ct",
            linked_segmentation=None,
        )
        google = self._properties_from_url(vs.to_url(target_site="google"))
        assert google[0]["type"] == "uint8"


class TestToolBindingsFollowCapabilities:
    """Bool properties and the tools that toggle them landed separately upstream."""

    def _layer(self, capabilities):
        df = pd.DataFrame({"x": [1], "y": [2], "z": [3], "ct": ["axon"]})
        vs = ViewerState(dimensions=[1, 1, 1], capabilities=capabilities)
        vs.add_points(
            df,
            point_column=["x", "y", "z"],
            tag_column="ct",
            linked_segmentation=None,
        )
        return vs.to_dict()["layers"][0]

    def test_properties_without_tools_bind_nothing(self):
        """A build between the two landing dates parses the properties but not the tools.

        Binding a tool type such a viewer cannot parse would cost the whole layer, so
        the properties still ship and the bindings are dropped.
        """
        from nglui.statebuilder.capabilities import (
            ANNOTATION_BOOL_PROPERTIES,
            Capabilities,
        )

        layer = self._layer(Capabilities.from_names([ANNOTATION_BOOL_PROPERTIES]))
        assert layer["annotationProperties"] == [{"id": "axon", "type": "bool"}]
        assert layer.get("toolBindings", {}) == {}

    def test_full_modern_binds_the_property_tools(self):
        layer = self._layer("main")
        assert layer["toolBindings"] == {
            "Q": {"type": "toggleBoolProperty", "property": "axon"}
        }

    def test_legacy_binds_tag_tools(self):
        layer = self._layer("legacy")
        assert layer["toolBindings"] == {"Q": "tagTool_tag0"}


class TestRepeatedTags:
    """A repeated tag cannot be represented on the wire under either encoding.

    Tags discovered from a dataframe are already distinct, but `tags=` is a public
    argument that accepts whatever it is given. Left alone, the bool encoding emitted
    two properties sharing an identifier (which Neuroglancer rejects outright) and the
    legacy encoding emitted a props vector shorter than its property list.
    """

    def _layer(self, capabilities):
        vs = ViewerState(dimensions=[1, 1, 1], capabilities=capabilities)
        with pytest.warns(UserWarning, match="repeated tags"):
            vs.add_annotation_layer(
                name="annos",
                resolution=[1, 1, 1],
                tags=["axon", "axon", "soma"],
                linked_segmentation=None,
            )
            vs.add_points(
                pd.DataFrame({"x": [1], "y": [1], "z": [1]}),
                name="annos",
                point_column=["x", "y", "z"],
            )
            return vs.to_dict()["layers"][0]

    @pytest.mark.parametrize("capabilities", ["main", "legacy"])
    def test_ids_stay_unique(self, capabilities):
        ids = [p["id"] for p in self._layer(capabilities)["annotationProperties"]]
        assert len(ids) == len(set(ids)) == 2

    @pytest.mark.parametrize("capabilities", ["main", "legacy"])
    def test_props_still_match_the_property_count(self, capabilities):
        layer = self._layer(capabilities)
        n = len(layer["annotationProperties"])
        assert all(len(a["props"]) == n for a in layer["annotations"])

    def test_order_of_first_appearance_is_kept(self):
        from nglui.statebuilder.ngl_annotations import unique_tags

        assert unique_tags(["soma", "axon", "soma", "glia"]) == ["soma", "axon", "glia"]
        assert unique_tags([]) == []
        assert unique_tags(None) == []


class TestTagLimitMessage:
    def test_names_the_encoding_and_the_way_out(self):
        """The ten-tag ceiling belongs to the Spelunker encoding, not to nglui."""
        vs = ViewerState(dimensions=[1, 1, 1], capabilities="legacy")
        n = 15
        vs.add_points(
            pd.DataFrame(
                {
                    "x": range(n),
                    "y": range(n),
                    "z": range(n),
                    "ct": [f"t{i}" for i in range(n)],
                }
            ),
            point_column=["x", "y", "z"],
            tag_column="ct",
            linked_segmentation=None,
        )
        with pytest.raises(ValueError, match="Spelunker tag encoding") as excinfo:
            vs.to_dict()
        assert "capabilities='main'" in str(excinfo.value)
