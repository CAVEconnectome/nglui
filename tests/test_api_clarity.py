"""API behaviors a notebook user relies on, from the viewer-options review."""

import pytest

from nglui.statebuilder import (
    AnnotationLayer,
    Camera,
    Panel,
    SegmentationLayer,
    SidePanel,
    SkeletonRendering,
    ViewerState,
    tools,
)

SEG = "precomputed://gs://example/seg"


def _vs(*layers, **kwargs):
    kwargs.setdefault("dimensions", [4, 4, 40])
    kwargs.setdefault("infer_coordinates", False)
    return ViewerState(layers=list(layers), **kwargs)


class TestSelectedLayer:
    def test_add_layer_selected_opens_the_panel(self):
        vs = _vs().add_layer(SegmentationLayer(name="s", source=SEG), selected=True)
        assert vs.to_dict()["selectedLayer"] == {"layer": "s", "visible": True}

    def test_constructor_selection_opens_the_panel(self):
        vs = _vs(SegmentationLayer(name="s", source=SEG), selected_layer="s")
        assert vs.to_dict()["selectedLayer"]["visible"] is True

    def test_set_selected_layer_can_leave_it_closed(self):
        vs = _vs(SegmentationLayer(name="s", source=SEG))
        vs.set_selected_layer("s", visible=False)
        assert vs.to_dict()["selectedLayer"] == {"layer": "s", "visible": False}

    def test_set_selected_layer_accepts_layer_object(self):
        seg = SegmentationLayer(name="s", source=SEG)
        vs = _vs(seg).set_selected_layer(seg)
        assert vs.to_dict()["selectedLayer"]["layer"] == "s"

    def test_unknown_selected_layer_raises(self):
        vs = _vs(SegmentationLayer(name="s", source=SEG)).set_selected_layer("nope")
        with pytest.raises(ValueError, match="'nope'"):
            vs.to_dict()

    def test_open_panel_without_a_layer_raises(self):
        vs = _vs(SegmentationLayer(name="s", source=SEG))
        vs.set_panels(selected_layer_panel=True)
        with pytest.raises(ValueError, match="set_selected_layer"):
            vs.to_dict()

    def test_panel_placement_combines_with_selection(self):
        vs = _vs(SegmentationLayer(name="s", source=SEG)).set_selected_layer("s")
        vs.set_panels(selected_layer_panel=SidePanel(side="left", size=300))
        assert vs.to_dict()["selectedLayer"] == {
            "layer": "s",
            "visible": True,
            "side": "left",
            "size": 300,
        }


class TestScaleCameraConflict:
    def test_conflicting_values_raise(self):
        with pytest.raises(ValueError, match="scale_3d"):
            _vs(scale_3d=10, camera=Camera(projection_scale=99))

    def test_agreeing_values_are_fine(self):
        vs = _vs(
            scale_3d=10, camera=Camera(projection_scale=10, projection_orientation="xz")
        )
        assert vs.to_dict()["projectionScale"] == 10.0


class TestSegmentationShaderControls:
    def test_forwarded_to_skeleton_rendering(self):
        layer = SegmentationLayer(source=SEG, shader_controls={"width": 2})
        assert layer.to_dict()["skeletonRendering"] == {"shaderControls": {"width": 2}}

    def test_combines_with_skeleton_rendering(self):
        layer = SegmentationLayer(
            source=SEG,
            shader_controls={"width": 2},
            skeleton_rendering=SkeletonRendering(line_width_3d=3),
        )
        assert layer.to_dict()["skeletonRendering"] == {
            "shaderControls": {"width": 2},
            "lineWidth3d": 3.0,
        }

    def test_conflict_raises(self):
        layer = SegmentationLayer(
            source=SEG,
            shader_controls={"width": 2},
            skeleton_rendering=SkeletonRendering(shader_controls={"width": 5}),
        )
        with pytest.raises(ValueError, match="shader_controls"):
            layer.to_dict()


class TestReadableErrors:
    @pytest.mark.parametrize(
        "make, fragment",
        [
            (lambda: SidePanel(side="rigth"), "side must be one of"),
            (lambda: SkeletonRendering(mode_3d="points"), "mode_3d must be one of"),
            (lambda: Panel(layout="4pannel"), "layout must be one of"),
            (lambda: tools.ToolPalette("P", side="up"), "side must be one of"),
        ],
    )
    def test_message_is_plain(self, make, fragment):
        with pytest.raises(ValueError) as err:
            make()
        assert fragment in str(err.value)
        assert "Attribute(" not in str(err.value)

    def test_active_tool_suggests_a_fix(self):
        with pytest.raises(ValueError, match="Did you mean 'point'"):
            AnnotationLayer(resolution=[4, 4, 40], active_tool="points")

    def test_side_suggests_a_fix(self):
        with pytest.raises(ValueError, match="Did you mean 'right'"):
            SidePanel(side="rigth")


class TestColors:
    def test_rgb_over_one_raises(self):
        with pytest.raises(ValueError, match="0 and 1"):
            _vs(projection_background_color=(128, 128, 128))

    @pytest.mark.parametrize(
        "color, expected",
        [((1, 0, 0), "#ff0000"), ((0.5, 0.5, 0.5), "#7f7f7f"), ("white", "#ffffff")],
    )
    def test_valid_colors(self, color, expected):
        d = _vs(projection_background_color=color).to_dict()
        assert d["projectionBackgroundColor"] == expected
