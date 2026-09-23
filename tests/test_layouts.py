"""Multi-panel layouts: rows, columns, and panels with their own layers and camera."""

import neuroglancer
import numpy as np
import pytest

from nglui.statebuilder import (
    Camera,
    ImageLayer,
    Panel,
    SegmentationLayer,
    StackLayout,
    ViewerState,
    column,
    row,
)
from nglui.statebuilder.viewer_config import NAMED_ORIENTATIONS

IMG_SRC = "precomputed://gs://example/img"
SEG_SRC = "precomputed://gs://example/seg"


def _state(**kwargs):
    return ViewerState(
        layers=[ImageLayer(source=IMG_SRC), SegmentationLayer(source=SEG_SRC)],
        dimensions=[4, 4, 40],
        infer_coordinates=False,
        **kwargs,
    )


class TestPanel:
    def test_basic(self):
        assert Panel(["img"], layout="3d").to_json(["img", "seg"]) == {
            "type": "viewer",
            "layers": ["img"],
            "layout": "3d",
        }

    def test_default_shows_all_layers(self):
        assert Panel().to_json(["img", "seg"])["layers"] == ["img", "seg"]

    def test_accepts_layer_objects_and_single_layer(self):
        seg = SegmentationLayer(name="cells", source=SEG_SRC)
        assert Panel(seg).layers == ["cells"]
        assert Panel([seg, "img"]).layers == ["cells", "img"]

    def test_camera_fields_are_unlinked(self):
        d = Panel(
            ["seg"],
            layout="3d",
            camera=Camera(projection_scale=500, projection_orientation="xz"),
        ).to_json(["seg"])
        assert d["projectionScale"] == {"link": "unlinked", "value": 500.0}
        assert d["projectionOrientation"]["link"] == "unlinked"
        assert "crossSectionScale" not in d

    def test_relative_camera(self):
        d = Panel(
            camera=Camera(projection_orientation="yz"), camera_link="relative"
        ).to_json(["seg"])
        assert d["projectionOrientation"]["link"] == "relative"

    @pytest.mark.parametrize(
        "kwargs",
        [{"layout": "row"}, {"camera_link": "linked"}, {"flex": 0}],
    )
    def test_invalid(self, kwargs):
        with pytest.raises(ValueError):
            Panel(**kwargs)


class TestStackLayout:
    def test_row_and_column_nest(self):
        layout = row(Panel(["img"]), column(Panel(["seg"]), Panel(["img"], flex=2)))
        d = layout.to_json(["img", "seg"])
        assert d["type"] == "row"
        assert d["children"][1]["type"] == "column"
        assert d["children"][1]["children"][1]["flex"] == 2

    def test_empty_is_rejected(self):
        with pytest.raises(ValueError, match="at least one child"):
            row()

    def test_preset_child_is_rejected(self):
        with pytest.raises(TypeError, match="Wrap a preset"):
            row("xy", Panel())

    def test_is_a_stack_layout(self):
        assert isinstance(column(Panel()), StackLayout)


class TestViewerStateLayout:
    def test_preset_still_works(self):
        assert _state(layout="4panel").to_dict()["layout"] == "4panel"

    def test_invalid_preset(self):
        with pytest.raises(ValueError, match="Invalid layout"):
            _state(layout="5panel")

    def test_set_layout(self):
        vs = _state().set_layout(
            row(Panel(["img", "seg"]), Panel(["seg"], layout="3d"))
        )
        d = vs.to_dict()["layout"]
        assert [c["layers"] for c in d["children"]] == [["img", "seg"], ["seg"]]

    def test_single_panel_layout(self):
        d = _state(layout=Panel(["seg"], layout="3d")).to_dict()["layout"]
        assert d == {"type": "viewer", "layers": ["seg"], "layout": "3d"}

    def test_unknown_layer_raises(self):
        vs = _state().set_layout(row(Panel(["nope"])))
        with pytest.raises(ValueError, match="not in the viewer"):
            vs.to_dict()

    def test_base_state_layers_count(self):
        base = {"layers": [{"type": "image", "name": "old", "source": IMG_SRC}]}
        vs = _state(base_state=base).set_layout(row(Panel(["old"]), Panel(["seg"])))
        assert vs.to_dict()["layout"]["children"][0]["layers"] == ["old"]

    def test_set_viewer_properties_layout(self):
        vs = _state().set_viewer_properties(layout=column(Panel(), Panel()))
        assert vs.to_dict()["layout"]["type"] == "column"


class TestComparisonWorkflow:
    """Two neurons side by side, each in its own 3d panel, with the EM below."""

    def test_round_trip(self):
        a, b = 864691135368056201, 864691135368056202
        vs = (
            ViewerState(dimensions=[4, 4, 40], infer_coordinates=False)
            .add_layer(ImageLayer(source=IMG_SRC))
            .add_layer(SegmentationLayer(name="cell_a", source=SEG_SRC, segments=[a]))
            .add_layer(SegmentationLayer(name="cell_b", source=SEG_SRC, segments=[b]))
            .set_layout(
                column(
                    row(
                        Panel(["cell_a"], layout="3d"),
                        Panel(
                            ["cell_b"],
                            layout="3d",
                            camera=Camera(projection_orientation="xz"),
                        ),
                        flex=2,
                    ),
                    Panel(["img", "cell_a", "cell_b"], layout="xy"),
                )
            )
        )
        url = vs.to_url(target_url="https://neuroglancer-demo.appspot.com")
        layout = neuroglancer.parse_url(url).to_json()["layout"]

        assert layout["type"] == "column"
        top, bottom = layout["children"]
        assert top["flex"] == 2
        assert [p["layers"] for p in top["children"]] == [["cell_a"], ["cell_b"]]
        right = top["children"][1]
        assert right["projectionOrientation"]["link"] == "unlinked"
        assert np.allclose(
            right["projectionOrientation"]["value"], NAMED_ORIENTATIONS["xz"], atol=1e-6
        )
        assert bottom["layers"] == ["img", "cell_a", "cell_b"]
