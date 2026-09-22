"""Viewer presentation, camera, side panels, and the raw-JSON `extra` passthrough."""

import neuroglancer
import numpy as np
import pytest

from nglui.statebuilder import (
    AnnotationLayer,
    Camera,
    ImageLayer,
    PointAnnotation,
    SegmentationLayer,
    SidePanel,
    ViewerState,
)
from nglui.statebuilder.utils import deep_merge
from nglui.statebuilder.viewer_config import NAMED_ORIENTATIONS, parse_orientation

IMG_SRC = "precomputed://gs://example/img"
SEG_SRC = "precomputed://gs://example/seg"


def _state(*layers, **kwargs):
    kwargs.setdefault("dimensions", [4, 4, 40])
    kwargs.setdefault("infer_coordinates", False)
    return ViewerState(layers=list(layers), **kwargs)


class TestParseOrientation:
    @pytest.mark.parametrize("name", ["xy", "xz", "yz"])
    def test_named(self, name):
        assert parse_orientation(name) == NAMED_ORIENTATIONS[name]

    def test_named_is_a_copy(self):
        parse_orientation("xy")[0] = 99
        assert NAMED_ORIENTATIONS["xy"][0] == 0.0

    def test_quaternion_is_normalized(self):
        assert parse_orientation([0, 0, 0, 2]) == [0.0, 0.0, 0.0, 1.0]
        q = parse_orientation(np.array([1, 1, 1, 1]))
        assert np.allclose(q, [0.5] * 4)

    def test_none(self):
        assert parse_orientation(None) is None

    @pytest.mark.parametrize("bad", ["front", [0, 0, 1], [0, 0, 0, 0]])
    def test_invalid(self, bad):
        with pytest.raises(ValueError):
            parse_orientation(bad)


class TestCamera:
    def test_unset_fields_are_none(self):
        cam = Camera()
        assert cam.projection_scale is None
        assert cam.cross_section_orientation is None

    def test_merge_keeps_unset_fields(self):
        cam = Camera(projection_scale=10, cross_section_depth=5).merge(
            Camera(projection_scale=20)
        )
        assert cam.projection_scale == 20.0
        assert cam.cross_section_depth == 5.0

    @pytest.mark.parametrize(
        "field, value, key",
        [
            ("cross_section_scale", 2.0, "crossSectionScale"),
            ("cross_section_depth", 10.0, "crossSectionDepth"),
            ("projection_scale", 5000.0, "projectionScale"),
            ("projection_depth", 3.0, "projectionDepth"),
        ],
    )
    def test_scalar_fields_emitted(self, field, value, key):
        vs = _state().set_camera(**{field: value})
        assert vs.to_dict()[key] == value

    @pytest.mark.parametrize(
        "field, key",
        [
            ("cross_section_orientation", "crossSectionOrientation"),
            ("projection_orientation", "projectionOrientation"),
        ],
    )
    def test_orientation_emitted(self, field, key):
        vs = _state().set_camera(**{field: "xz"})
        assert np.allclose(vs.to_dict()[key], NAMED_ORIENTATIONS["xz"], atol=1e-6)

    def test_scale_properties_are_camera_views(self):
        vs = _state(scale_imagery=3.0)
        assert vs.camera.cross_section_scale == 3.0
        vs.set_camera(projection_scale=123)
        assert vs.scale_3d == 123.0
        vs.scale_imagery = 7
        assert vs.camera.cross_section_scale == 7.0

    def test_set_camera_accepts_camera_object(self):
        side = Camera(projection_orientation="yz", projection_scale=800)
        d = _state().set_camera(side).to_dict()
        assert d["projectionScale"] == 800.0
        assert "projectionOrientation" in d

    def test_constructor_camera_overrides_flat_scales(self):
        vs = _state(scale_3d=100, camera=Camera(projection_scale=200))
        assert vs.to_dict()["projectionScale"] == 200.0

    def test_camera_respects_base_state(self):
        base = {"projectionScale": 1234.0, "crossSectionScale": 5.0}
        d = _state(base_state=base).set_camera(projection_orientation="xz").to_dict()
        assert d["projectionScale"] == 1234.0
        assert d["crossSectionScale"] == 5.0
        assert "projectionOrientation" in d


class TestDisplayOptions:
    @pytest.mark.parametrize(
        "name, value, key, expected",
        [
            ("title", "My neuron", "title", "My neuron"),
            ("show_axis_lines", False, "showAxisLines", False),
            ("show_scale_bar", False, "showScaleBar", False),
            ("show_default_annotations", False, "showDefaultAnnotations", False),
            (
                "cross_section_background_color",
                "white",
                "crossSectionBackgroundColor",
                "#ffffff",
            ),
            (
                "projection_background_color",
                (0, 0, 0),
                "projectionBackgroundColor",
                "#000000",
            ),
            (
                "hide_cross_section_background_3d",
                True,
                "hideCrossSectionBackground3D",
                True,
            ),
            ("wire_frame", True, "wireFrame", True),
        ],
    )
    def test_option_emitted(self, name, value, key, expected):
        from_init = _state(**{name: value}).to_dict()
        assert from_init[key] == expected
        from_setter = _state().set_viewer_properties(**{name: value}).to_dict()
        assert from_setter[key] == expected
        vs = _state()
        setattr(vs, name, value)
        assert vs.to_dict()[key] == expected

    def test_unset_options_are_omitted(self):
        d = _state().to_dict()
        for key in ["title", "showAxisLines", "showScaleBar", "wireFrame"]:
            assert key not in d

    def test_property_reads_and_unsets(self):
        vs = _state(title="a")
        assert vs.title == "a"
        vs.title = None
        assert vs.title is None
        assert "title" not in vs.to_dict()

    def test_set_viewer_properties_leaves_others(self):
        vs = _state(title="a", show_scale_bar=False)
        vs.set_viewer_properties(wire_frame=True)
        d = vs.to_dict()
        assert d["title"] == "a"
        assert d["showScaleBar"] is False
        assert d["wireFrame"] is True

    def test_overrides_base_state(self):
        d = _state(base_state={"title": "old"}, title="new").to_dict()
        assert d["title"] == "new"


class TestPanels:
    def test_bool_shorthand(self):
        d = _state().set_panels(layer_list=True, statistics=False).to_dict()
        assert d["layerListPanel"] == {"visible": True}
        assert d["statistics"] == {"visible": False}

    def test_side_panel_fields(self):
        d = (
            _state()
            .set_panels(help=SidePanel(visible=True, side="left", size=300))
            .to_dict()
        )
        assert d["helpPanel"] == {"visible": True, "side": "left", "size": 300}

    def test_dict_is_accepted(self):
        d = _state().set_panels(layer_list={"side": "right"}).to_dict()
        assert d["layerListPanel"] == {"side": "right"}

    def test_repeat_calls_merge(self):
        vs = _state().set_panels(layer_list=SidePanel(side="left"))
        vs.set_panels(layer_list=True)
        assert vs.to_dict()["layerListPanel"] == {"visible": True, "side": "left"}

    def test_selected_layer_panel_combines_with_layer(self):
        vs = _state(ImageLayer(source=IMG_SRC), selected_layer="img")
        vs.set_panels(selected_layer=SidePanel(visible=True, side="right", size=500))
        assert vs.to_dict()["selectedLayer"] == {
            "layer": "img",
            "visible": True,
            "side": "right",
            "size": 500,
        }

    def test_invalid_side(self):
        with pytest.raises(ValueError):
            SidePanel(side="middle")

    def test_invalid_type(self):
        with pytest.raises(TypeError):
            _state().set_panels(help="yes")


class TestDeepMerge:
    def test_nested_merge_and_removal(self):
        base = {"a": {"b": 1, "c": 2}, "d": 3}
        assert deep_merge(base, {"a": {"b": 5}, "d": None}) == {"a": {"b": 5, "c": 2}}
        assert base == {"a": {"b": 1, "c": 2}, "d": 3}

    def test_lists_are_replaced(self):
        assert deep_merge({"a": [1, 2, 3]}, {"a": [9]}) == {"a": [9]}


class TestExtra:
    def test_viewer_extra_adds_unwrapped_fields(self):
        d = _state(extra={"gpuMemoryLimit": 2_000_000_000}).to_dict()
        assert d["gpuMemoryLimit"] == 2_000_000_000

    def test_viewer_extra_wins_and_removes(self):
        d = _state(title="a", extra={"title": "b", "showSlices": None}).to_dict()
        assert d["title"] == "b"
        assert "showSlices" not in d

    def test_viewer_extra_keeps_layers(self):
        vs = _state(ImageLayer(source=IMG_SRC), extra={"concurrentDownloads": 64})
        d = vs.to_dict()
        assert d["concurrentDownloads"] == 64
        assert [layer["name"] for layer in d["layers"]] == ["img"]

    def test_set_viewer_properties_merges_extra(self):
        vs = _state(extra={"a": {"x": 1}})
        vs.set_viewer_properties(extra={"a": {"y": 2}})
        assert vs.extra == {"a": {"x": 1, "y": 2}}

    def test_layer_extra_in_layer_dict(self):
        layer = SegmentationLayer(source=SEG_SRC, extra={"saturation": 0.5})
        assert layer.to_dict()["saturation"] == 0.5

    def test_layer_extra_in_state(self):
        layer = ImageLayer(source=IMG_SRC, extra={"opacity": 0.25, "madeUp": 1})
        d = _state(layer).to_dict()["layers"][0]
        assert d["opacity"] == 0.25
        assert d["madeUp"] == 1

    def test_layer_extra_on_local_annotations(self):
        layer = AnnotationLayer(
            resolution=[4, 4, 40],
            annotations=[PointAnnotation(point=[1, 2, 3])],
            set_position=False,
            extra={"annotationColor": "#123456"},
        )
        d = _state(layer).to_dict()["layers"][0]
        assert d["annotationColor"] == "#123456"
        assert len(d["annotations"]) == 1


class TestFigureWorkflow:
    """A realistic state for a figure, round-tripped through a URL."""

    def test_full_state_round_trips(self):
        vs = (
            ViewerState(
                dimensions=[4, 4, 40],
                infer_coordinates=False,
                title="Pyramidal cell 864691135",
                show_axis_lines=False,
                show_scale_bar=False,
                projection_background_color="white",
                layout="3d",
                extra={"gpuMemoryLimit": 4_000_000_000},
            )
            .add_layer(ImageLayer(source=IMG_SRC, opacity=0.8))
            .add_layer(
                SegmentationLayer(
                    source=SEG_SRC,
                    segments=[864691135],
                    hide_segment_zero=True,
                    extra={"meshRenderScale": 1},
                ),
                selected=True,
            )
            .set_camera(projection_orientation="xz", projection_scale=12000)
            .set_panels(layer_list=True, selected_layer=SidePanel(side="right"))
        )
        url = vs.to_url(target_url="https://neuroglancer-demo.appspot.com")
        parsed = neuroglancer.parse_url(url).to_json()

        assert parsed["title"] == "Pyramidal cell 864691135"
        assert parsed["showAxisLines"] is False
        assert parsed["showScaleBar"] is False
        assert parsed["projectionBackgroundColor"] == "#ffffff"
        assert parsed["layout"] == "3d"
        assert parsed["projectionScale"] == 12000
        assert np.allclose(
            parsed["projectionOrientation"], NAMED_ORIENTATIONS["xz"], atol=1e-6
        )
        assert parsed["gpuMemoryLimit"] == 4_000_000_000
        assert parsed["layerListPanel"] == {"visible": True}
        assert parsed["selectedLayer"]["layer"] == "seg"
        assert parsed["selectedLayer"]["side"] == "right"
        img, seg = parsed["layers"]
        assert img["opacity"] == 0.8
        assert seg["segments"] == ["864691135"]
        assert seg["meshRenderScale"] == 1
