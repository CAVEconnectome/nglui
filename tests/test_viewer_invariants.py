"""Validate emitted states against Neuroglancer's own parsing rules.

These are the constraints the viewer enforces when it loads a state. Violating any
of them fails the whole annotation layer, not just one property, and the Python
neuroglancer library checks none of them -- so they are asserted here instead.

The rules are taken from the deployed client at neuroglancer-demo.appspot.com:
`parseAnnotationPropertyId` (/^[a-z][a-zA-Z0-9_]*$/),
`ensureUniqueAnnotationPropertyIds`, `parseAnnotationPropertyType`, and the
`expectArray(propsObj, propSpecs.length)` in `parseAnnotationPropertySpec`.
"""

import re
import warnings

import pandas as pd
import pytest

from nglui.statebuilder import ViewerState

PROPERTY_ID_RE = re.compile(r"^[a-z][a-zA-Z0-9_]*$")

# propertyTypeDataType in the deployed client, which throws on anything else.
VALID_PROPERTY_TYPES = {
    "rgb",
    "rgba",
    "bool",
    "float32",
    "uint32",
    "int32",
    "uint16",
    "int16",
    "uint8",
    "int8",
}

AWKWARD_LABELS = [
    "Cell Body",
    "post-synaptic",
    "1st pass",
    "UPPER",
    "with.dots",
    "trailing ",
    "emoji \U0001f9e0 tag",
    "a" * 80,
]


def assert_layer_is_loadable(layer):
    """Assert a serialized annotation layer satisfies Neuroglancer's parse rules."""
    specs = layer.get("annotationProperties", [])

    ids = [spec["id"] for spec in specs]
    for spec_id in ids:
        assert PROPERTY_ID_RE.match(spec_id), f"invalid property id {spec_id!r}"
    assert len(ids) == len(set(ids)), f"duplicate property ids in {ids}"

    for spec in specs:
        assert spec["type"] in VALID_PROPERTY_TYPES, f"unknown type {spec['type']!r}"
        if "enum_values" in spec:
            assert "enum_labels" in spec
            assert len(spec["enum_values"]) == len(spec["enum_labels"])

    for anno in layer.get("annotations", []):
        props = anno.get("props", [])
        assert len(props) == len(
            specs
        ), f"props length {len(props)} does not match {len(specs)} properties"


def _build(tags, capabilities):
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4], "z": [5, 6]})
    for i, tag in enumerate(tags):
        df[tag] = [i % 2 == 0, i % 2 == 1]
    vs = ViewerState(dimensions=[1, 1, 1], capabilities=capabilities)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vs.add_points(
            df,
            point_column=["x", "y", "z"],
            tag_bools=list(tags),
            linked_segmentation=None,
        )
        return vs.to_dict()["layers"][0]


class TestEmittedStatesAreLoadable:
    @pytest.mark.parametrize("capabilities", ["main", "legacy"])
    def test_simple_tags(self, capabilities):
        assert_layer_is_loadable(_build(["axon", "soma"], capabilities))

    @pytest.mark.parametrize("label", AWKWARD_LABELS)
    def test_awkward_labels_still_produce_a_loadable_layer(self, label):
        """Tag labels come from dataframe columns and are entirely uncontrolled."""
        assert_layer_is_loadable(_build([label], "main"))

    def test_all_awkward_labels_together(self):
        """Sanitizing many odd labels at once must not collide them."""
        assert_layer_is_loadable(_build(AWKWARD_LABELS, "main"))

    def test_labels_that_sanitize_to_the_same_id(self):
        assert_layer_is_loadable(
            _build(["Cell Body", "cell body", "CELL-BODY"], "main")
        )

    def test_legacy_ids_are_also_valid(self):
        """`tag0` happens to satisfy the modern id rule, so legacy states still load."""
        assert_layer_is_loadable(_build(AWKWARD_LABELS[:3], "legacy"))

    def test_no_tags(self):
        df = pd.DataFrame({"x": [1], "y": [2], "z": [3]})
        vs = ViewerState(dimensions=[1, 1, 1], capabilities="main")
        vs.add_points(df, point_column=["x", "y", "z"], linked_segmentation=None)
        assert_layer_is_loadable(vs.to_dict()["layers"][0])


class TestToolBindingsAreLoadable:
    def test_binding_keys_are_single_capitals(self):
        """Neuroglancer's ToolBindings validates keys against ^[A-Z]$."""
        for capabilities in ("main", "legacy"):
            layer = _build(["axon", "soma"], capabilities)
            for key in layer.get("toolBindings", {}):
                assert re.match(r"^[A-Z]$", key), f"invalid binding key {key!r}"

    def test_bindings_reference_declared_properties(self):
        layer = _build(["axon", "soma"], "main")
        declared = {spec["id"] for spec in layer["annotationProperties"]}
        for binding in layer["toolBindings"].values():
            assert binding["property"] in declared
