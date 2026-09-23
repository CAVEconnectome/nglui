"""Regression tests for problems found auditing the viewer-options branch."""

import json
import warnings

import attrs
import numpy as np
import pytest

from nglui.statebuilder import (
    AnnotationLayer,
    Camera,
    ImageLayer,
    SegmentationLayer,
    SidePanel,
    ViewerState,
    tools,
)

SEG = "precomputed://gs://example/seg"
IMG = "precomputed://gs://example/img"


def _vs(*layers, **kwargs):
    kwargs.setdefault("dimensions", [4, 4, 40])
    kwargs.setdefault("infer_coordinates", False)
    return ViewerState(layers=list(layers), **kwargs)


def _anno(name, n_tags):
    return AnnotationLayer(
        name=name,
        resolution=[4, 4, 40],
        tags=[f"t{i}" for i in range(n_tags)],
        set_position=False,
    )


def _bindings(d, name):
    return next(layer for layer in d["layers"] if layer["name"] == name).get(
        "toolBindings", {}
    )


class TestTagKeysRunOut:
    def test_leftover_tag_tools_are_dropped_with_a_warning(self):
        vs = _vs(*(_anno(n, 10) for n in "abc"), capabilities="main")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            d = vs.to_dict()
        messages = [str(w.message) for w in caught]
        assert any("No free key" in m and "layer 'c'" in m for m in messages)
        keys = [k for n in "abc" for k in _bindings(d, n)]
        assert len(keys) == len(set(keys)) == 26
        assert len(_bindings(d, "a")) == 10  # the first layer keeps all its tags

    def test_requested_tools_still_raise_when_out_of_keys(self):
        vs = _vs(*(_anno(n, 10) for n in "abc"), capabilities="main")
        vs.add_tools(tools.Dimension(dimension="z"))
        with pytest.raises(ValueError, match="Every key"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                vs.to_dict()


class TestTagRelocation:
    def test_only_the_conflicting_tag_moves(self):
        vs = _vs(
            SegmentationLayer(name="seg", source=SEG),
            _anno("a", 2),
            capabilities="main",
        )
        vs.add_tools(tools.SelectSegments(layer="seg", key="Q"))
        with pytest.warns(UserWarning, match="moved"):
            d = vs.to_dict()
        bindings = _bindings(d, "a")
        # t1 keeps its usual W; only t0 moves off the Q the user claimed
        assert bindings["W"]["property"] == "t1"
        assert bindings["E"]["property"] == "t0"

    def test_moves_are_reported(self):
        vs = _vs(_anno("a", 1), _anno("b", 1), capabilities="main")
        with pytest.warns(UserWarning, match="layer 'b'"):
            vs.to_dict()

    def test_no_warning_when_nothing_moves(self):
        vs = _vs(_anno("a", 2), capabilities="main")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            vs.to_dict()


class TestPaletteReuse:
    def test_reusing_a_palette_object_does_not_duplicate(self):
        palette = tools.ToolPalette("P")
        vs = _vs(SegmentationLayer(name="seg", source=SEG))
        vs.add_tools(tools.SelectSegments(layer="seg"), palette=palette)
        vs.add_tools(tools.MeshSilhouette(layer="seg"), palette=palette)
        entries = vs.to_dict()["toolPalettes"]["P"]["tools"]
        assert [e["type"] for e in entries] == [
            "selectSegments",
            "meshSilhouetteRendering",
        ]
        assert palette.tools == []  # the caller's object is untouched

    def test_palette_does_not_leak_between_states(self):
        palette = tools.ToolPalette("P")
        a = _vs(SegmentationLayer(name="seg", source=SEG))
        b = _vs(ImageLayer(name="img", source=IMG))
        a.add_tools(tools.SelectSegments(layer="seg"), palette=palette)
        b.add_tools(tools.Opacity(layer="img"), palette=palette)
        assert [e["type"] for e in a.to_dict()["toolPalettes"]["P"]["tools"]] == [
            "selectSegments"
        ]


class TestNoStaleState:
    def test_camera_is_immutable(self):
        vs = _vs()
        with pytest.raises(attrs.exceptions.FrozenInstanceError):
            vs.camera.projection_scale = 123

    @pytest.mark.parametrize(
        "obj",
        [
            Camera(),
            SidePanel(),
            tools.SelectSegments(),
            tools.ShaderControl(),
        ],
    )
    def test_config_objects_are_immutable(self, obj):
        field = attrs.fields(type(obj))[0].name
        with pytest.raises(attrs.exceptions.FrozenInstanceError):
            setattr(obj, field, None)

    def test_extra_cannot_be_mutated_in_place(self):
        vs = _vs(extra={"title": "one"})
        vs.to_dict()
        with pytest.raises(TypeError):
            vs.extra["title"] = "two"
        vs.extra = {"title": "two"}
        assert vs.to_dict()["title"] == "two"


class TestExtraNone:
    def test_set_viewer_properties_none_removes_key(self):
        vs = _vs().set_viewer_properties(extra={"showSlices": None})
        assert "showSlices" not in vs.to_dict()

    def test_later_value_overrides_earlier_none(self):
        vs = _vs(extra={"showSlices": None}).set_viewer_properties(
            extra={"showSlices": True}
        )
        assert vs.to_dict()["showSlices"] is True


class TestViewerExtraToolBindings:
    def test_extra_keys_are_reserved(self):
        vs = _vs(extra={"toolBindings": {"Z": {"type": "dimension", "dimension": "z"}}})
        vs.add_tools(tools.Dimension(dimension="y"))
        bindings = vs.to_dict()["toolBindings"]
        assert bindings["Z"]["dimension"] == "z"
        assert {"type": "dimension", "dimension": "y"} in bindings.values()

    def test_explicit_key_colliding_with_extra_raises(self):
        vs = _vs(extra={"toolBindings": {"Z": {"type": "dimension", "dimension": "z"}}})
        vs.add_tools(tools.Dimension(dimension="y", key="Z"))
        with pytest.raises(ValueError, match="already bound"):
            vs.to_dict()


class TestDisplayOptionTypes:
    @pytest.mark.parametrize(
        "name, value",
        [("show_axis_lines", "false"), ("wire_frame", 1), ("title", 5)],
    )
    def test_wrong_type_raises(self, name, value):
        with pytest.raises(TypeError, match=name):
            _vs(**{name: value})

    def test_numpy_bool_is_accepted(self):
        assert _vs(show_scale_bar=np.bool_(False)).to_dict()["showScaleBar"] is False


class TestPlainJson:
    def test_numpy_in_shader_controls_is_serializable(self):
        layer = ImageLayer(
            source=IMG, shader_controls={"normalized": {"range": np.array([1, 200])}}
        )
        json.dumps(_vs(layer).to_dict()["layers"][0]["shaderControls"])

    def test_numpy_in_extra_is_serializable(self):
        d = _vs(extra={"crossSectionOrientation": np.array([0.0, 0, 0, 1])}).to_dict()
        json.dumps(d["crossSectionOrientation"])
