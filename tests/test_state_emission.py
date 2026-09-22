"""Every option nglui accepts must reach the emitted state.

A setting that is stored but never serialized is worse than a missing one: the user
has no signal that it did nothing. These tests pin each accepted option to the JSON
key it is supposed to produce.
"""

import pandas as pd
import pytest

from nglui.statebuilder.base import ViewerState
from nglui.statebuilder.ngl_components import (
    AnnotationLayer,
    CoordSpaceTransform,
    DataMap,
    ImageLayer,
    PointAnnotation,
    SegmentationLayer,
    Source,
)

IMG_SRC = "precomputed://gs://example/img"
SEG_SRC = "precomputed://gs://example/seg"


def _state(*layers, **kwargs):
    kwargs.setdefault("dimensions", [4, 4, 40])
    kwargs.setdefault("infer_coordinates", False)
    return ViewerState(layers=list(layers), **kwargs)


class TestImageLayerEmission:
    @pytest.mark.parametrize(
        "attr, value, key",
        [
            ("opacity", 0.8, "opacity"),
            ("blend", "additive", "blend"),
            ("volume_rendering_mode", "on", "volumeRendering"),
            ("volume_rendering_gain", 2.5, "volumeRenderingGain"),
            ("volume_rendering_depth_samples", 128, "volumeRenderingDepthSamples"),
            ("cross_section_render_scale", 2.0, "crossSectionRenderScale"),
        ],
    )
    def test_option_is_emitted(self, attr, value, key):
        layer = ImageLayer(source=IMG_SRC, **{attr: value})
        assert layer.to_dict()[key] == value

    def test_unset_options_defer_to_neuroglancer(self):
        d = ImageLayer(source=IMG_SRC).to_dict()
        for key in [
            "opacity",
            "blend",
            "volumeRendering",
            "volumeRenderingGain",
            "volumeRenderingDepthSamples",
            "crossSectionRenderScale",
        ]:
            assert key not in d

    def test_options_emitted_with_shader(self):
        d = ImageLayer(source=IMG_SRC, shader="void main() {}", opacity=0.3).to_dict()
        assert d["shader"] == "void main() {}"
        assert d["opacity"] == 0.3


class TestSegmentationLayerEmission:
    def test_hide_segment_zero_false_is_emitted(self):
        d = SegmentationLayer(source=SEG_SRC, hide_segment_zero=False).to_dict()
        assert d["hideSegmentZero"] is False

    def test_hide_segment_zero_unset_is_omitted(self):
        assert "hideSegmentZero" not in SegmentationLayer(source=SEG_SRC).to_dict()

    def test_set_view_options_accepts_zero(self):
        layer = SegmentationLayer(source=SEG_SRC).set_view_options(
            selected_alpha=0, not_selected_alpha=0, alpha_3d=0, mesh_silhouette=0
        )
        d = layer.to_dict()
        assert d["selectedAlpha"] == 0
        assert d["objectAlpha"] == 0
        assert layer.mesh_silhouette == 0

    def test_segments_from_data_via_datamap(self):
        df = pd.DataFrame({"root_id": [1, 2, 3], "color": ["red", "blue", "green"]})
        layer = SegmentationLayer(source=SEG_SRC).add_segments_from_data(
            DataMap("segs"), segment_column="root_id", color_column="color"
        )
        mapped = layer.map({"segs": df})
        d = mapped.to_dict()
        assert sorted(d["segments"]) == ["1", "2", "3"]
        assert set(d["segmentColors"]) == {"1", "2", "3"}

    def test_segment_properties_via_datamap(self):
        df = pd.DataFrame({"pt_root_id": [1, 2], "cell_type": ["a", "b"]})
        layer = SegmentationLayer(source=SEG_SRC).add_segment_properties(
            DataMap("props"), client=None, label_column="cell_type", dry_run=True
        )
        # Registering must not eagerly process the DataMap placeholder
        assert not layer.is_static
        mapped = layer.map({"props": df})
        urls = [s["url"] for s in mapped.to_dict()["source"]]
        assert "DRYRUN_SEGMENT_PROPERTIES" in urls


class TestAnnotationLayerEmission:
    def _local(self, **kwargs):
        return AnnotationLayer(
            resolution=[4, 4, 40],
            linked_segmentation={"segments": "seg"},
            annotations=[PointAnnotation(point=[1, 2, 3], segments=[5])],
            set_position=False,
            **kwargs,
        )

    @pytest.mark.parametrize(
        "filter_value, expected",
        [
            (True, ["segments"]),
            (["segments"], ["segments"]),
            ("segments", ["segments"]),
        ],
    )
    def test_local_filter_by_segmentation(self, filter_value, expected):
        d = self._local(filter_by_segmentation=filter_value).to_dict()
        assert d["filterBySegmentation"] == expected

    def test_local_filter_unset_is_omitted(self):
        assert "filterBySegmentation" not in self._local().to_dict()

    @pytest.mark.parametrize(
        "filter_value, expected",
        [
            (True, ["segments"]),
            (["segments"], ["segments"]),
            ("segments", ["segments"]),
        ],
    )
    def test_cloud_filter_by_segmentation(self, filter_value, expected):
        layer = AnnotationLayer(
            source="precomputed://gs://example/anno",
            linked_segmentation={"segments": "seg"},
            filter_by_segmentation=filter_value,
        )
        assert layer.to_dict()["filterBySegmentation"] == expected


class TestCoordSpaceTransform:
    def test_input_dimensions_accepts_list(self):
        t = CoordSpaceTransform(
            output_dimensions=[4, 4, 40], input_dimensions=[8, 8, 40]
        ).to_neuroglancer()
        assert t.input_dimensions is not None

    def test_transform_round_trips_through_source(self):
        src = Source(
            url=IMG_SRC,
            transform=CoordSpaceTransform(
                output_dimensions=[4, 4, 40], input_dimensions=[8, 8, 40]
            ),
        )
        d = ImageLayer(source=src).to_dict()
        assert "inputDimensions" in d["source"][0]["transform"]


class TestSelectedLayer:
    def test_selected_layer_is_emitted(self):
        vs = _state(ImageLayer(source=IMG_SRC), selected_layer="img")
        assert vs.to_dict()["selectedLayer"]["layer"] == "img"

    def test_selected_layer_visible_is_emitted(self):
        vs = _state(
            ImageLayer(source=IMG_SRC),
            selected_layer="img",
            selected_layer_visible=True,
        )
        assert vs.to_dict()["selectedLayer"] == {"layer": "img", "visible": True}

    def test_add_layer_selected(self):
        vs = _state()
        vs.add_layer(ImageLayer(source=IMG_SRC), selected=True)
        assert vs.to_dict()["selectedLayer"]["layer"] == "img"

    def test_set_viewer_properties_selected_layer_visible(self):
        vs = _state(ImageLayer(source=IMG_SRC))
        vs.set_viewer_properties(selected_layer="img", selected_layer_visible=True)
        assert vs.to_dict()["selectedLayer"] == {"layer": "img", "visible": True}

    def test_unselected_state_has_no_selected_layer(self):
        assert "selectedLayer" not in _state(ImageLayer(source=IMG_SRC)).to_dict()


class TestViewerProperties:
    def test_set_viewer_properties_keeps_infer_coordinates(self):
        vs = ViewerState(infer_coordinates=True)
        vs.set_viewer_properties(position=[1, 2, 3])
        assert vs.infer_coordinates is True

    def test_defaults_applied_without_base_state(self):
        d = _state().to_dict()
        assert d["crossSectionScale"] == 1.0
        assert d["projectionScale"] == 50000.0
        assert d["showSlices"] is False
        assert d["layout"] == "xy-3d"

    def test_base_state_values_kept_when_not_overridden(self):
        base = {"crossSectionScale": 3.0, "projectionScale": 1234.0, "layout": "4panel"}
        d = _state(base_state=base).to_dict()
        assert d["crossSectionScale"] == 3.0
        assert d["projectionScale"] == 1234.0
        assert d["layout"] == "4panel"

    def test_explicit_values_override_base_state(self):
        base = {"crossSectionScale": 3.0, "projectionScale": 1234.0, "layout": "4panel"}
        vs = _state(base_state=base, scale_imagery=2.0, layout="3d")
        vs.scale_3d = 99.0
        vs.show_slices = True
        d = vs.to_dict()
        assert d["crossSectionScale"] == 2.0
        assert d["projectionScale"] == 99.0
        assert d["showSlices"] is True
        assert d["layout"] == "3d"
