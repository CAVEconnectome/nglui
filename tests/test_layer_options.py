"""Layer display options: shader controls, segmentation options, skeleton rendering."""

import urllib.parse

import neuroglancer
import pandas as pd
import pytest

from nglui.statebuilder import (
    AnnotationLayer,
    DataMap,
    ImageLayer,
    PointAnnotation,
    SegmentationLayer,
    SkeletonRendering,
    ViewerState,
)
from nglui.statebuilder.shaders import DEFAULT_SHADER_MAP

IMG_SRC = "precomputed://gs://example/img"
SEG_SRC = "precomputed://gs://example/seg"
ANNO_SRC = "precomputed://gs://example/anno"
BIG_ID = 864691135368056201  # a real-sized root id, beyond 2**53


class TestImageShaderControls:
    def test_shader_controls_emitted(self):
        layer = ImageLayer(
            source=IMG_SRC, shader_controls={"normalized": {"range": [10, 200]}}
        )
        assert layer.to_dict()["shaderControls"] == {"normalized": {"range": [10, 200]}}

    def test_unset_is_omitted(self):
        assert "shaderControls" not in ImageLayer(source=IMG_SRC).to_dict()

    def test_set_contrast_default_control(self):
        d = ImageLayer(source=IMG_SRC).set_contrast(range=(40, 210)).to_dict()
        assert d["shaderControls"] == {"normalized": {"range": [40.0, 210.0]}}

    def test_set_contrast_merges(self):
        layer = ImageLayer(source=IMG_SRC, shader_controls={"gain": 2})
        layer.set_contrast(range=(0, 100)).set_contrast(window=(0, 255))
        assert layer.shader_controls == {
            "gain": 2,
            "normalized": {"range": [0.0, 100.0], "window": [0.0, 255.0]},
        }

    def test_set_contrast_named_control(self):
        layer = ImageLayer(source=IMG_SRC).set_contrast(range=(1, 2), control="red")
        assert layer.shader_controls == {"red": {"range": [1.0, 2.0]}}

    def test_contrast_through_viewer_state(self):
        vs = ViewerState(dimensions=[4, 4, 40], infer_coordinates=False)
        vs.add_image_layer(IMG_SRC, shader_controls={"normalized": {"range": [5, 9]}})
        layer = vs.to_dict()["layers"][0]
        assert layer["shaderControls"] == {"normalized": {"range": [5, 9]}}


class TestSegmentationOptions:
    @pytest.mark.parametrize(
        "attr, value, key, expected",
        [
            ("segment_query", "#axon", "segmentQuery", "#axon"),
            ("saturation", 0.4, "saturation", 0.4),
            ("color_seed", 7, "colorSeed", 7),
            ("segment_default_color", "red", "segmentDefaultColor", "#ff0000"),
            ("mesh_render_scale", 2.0, "meshRenderScale", 2.0),
            ("cross_section_render_scale", 2.0, "crossSectionRenderScale", 2.0),
            ("hover_highlight", False, "hoverHighlight", False),
            ("base_segment_coloring", True, "baseSegmentColoring", True),
            ("ignore_null_visible_set", False, "ignoreNullVisibleSet", False),
            ("linked_segmentation_group", "seg", "linkedSegmentationGroup", "seg"),
            (
                "linked_segmentation_color_group",
                False,
                "linkedSegmentationColorGroup",
                False,
            ),
        ],
    )
    def test_option_emitted(self, attr, value, key, expected):
        d = SegmentationLayer(source=SEG_SRC, **{attr: value}).to_dict()
        assert d[key] == expected

    def test_unset_options_are_omitted(self):
        d = SegmentationLayer(source=SEG_SRC).to_dict()
        for key in [
            "segmentQuery",
            "saturation",
            "colorSeed",
            "segmentDefaultColor",
            "meshRenderScale",
            "equivalences",
            "skeletonRendering",
            "linkedSegmentationGroup",
        ]:
            assert key not in d

    def test_equivalences_keep_uint64_precision(self):
        # Neuroglancer parses states with JSON.parse, which rounds bare integers
        # past 2**53; real root ids must travel as strings.
        d = SegmentationLayer(
            source=SEG_SRC, equivalences=[[BIG_ID, BIG_ID + 1], [1, 2]]
        ).to_dict()
        assert d["equivalences"] == [[1, 2], [str(BIG_ID), str(BIG_ID + 1)]]

    def test_equivalences_in_url_are_strings(self):
        vs = ViewerState(dimensions=[4, 4, 40], infer_coordinates=False)
        vs.add_layer(SegmentationLayer(source=SEG_SRC, equivalences=[[BIG_ID, 5]]))
        url = urllib.parse.unquote(vs.to_url(target_url="https://example.org"))
        assert f'"{BIG_ID}"' in url
        parsed = neuroglancer.parse_url(vs.to_url(target_url="https://example.org"))
        assert parsed.layers["seg"].equivalences.to_json() == [[5, str(BIG_ID)]]

    def test_equivalences_from_base_state_are_safe(self):
        base = {
            "layers": [
                {
                    "type": "segmentation",
                    "name": "old",
                    "source": SEG_SRC,
                    "equivalences": [[str(BIG_ID), "7"]],
                }
            ]
        }
        vs = ViewerState(
            dimensions=[4, 4, 40], infer_coordinates=False, base_state=base
        )
        (group,) = vs.to_dict()["layers"][0]["equivalences"]
        assert str(BIG_ID) in group
        assert BIG_ID not in group

    def test_linked_skeleton_layer_workflow(self):
        """A skeleton-only layer following a mesh layer's selection and colors."""
        vs = ViewerState(dimensions=[4, 4, 40], infer_coordinates=False)
        vs.add_layer(SegmentationLayer(name="seg", source=SEG_SRC, segments=[BIG_ID]))
        vs.add_layer(
            SegmentationLayer(
                name="skel",
                source="precomputed://gs://example/skeletons",
                linked_segmentation_group="seg",
                skeleton_rendering=SkeletonRendering(line_width_3d=3),
            )
        )
        url = vs.to_url(target_url="https://neuroglancer-demo.appspot.com")
        seg, skel = neuroglancer.parse_url(url).to_json()["layers"]
        assert seg["segments"] == [str(BIG_ID)]
        assert skel["linkedSegmentationGroup"] == "seg"
        assert skel["skeletonRendering"] == {"lineWidth3d": 3}


class TestSkeletonRendering:
    def test_fields_map_to_json_keys(self):
        rendering = SkeletonRendering(
            shader="void main() {}",
            shader_controls={"width": 2},
            mode_2d="lines_and_points",
            mode_3d="lines",
            line_width_2d=1,
            line_width_3d=4,
        )
        d = SegmentationLayer(source=SEG_SRC, skeleton_rendering=rendering).to_dict()
        assert d["skeletonRendering"] == {
            "shader": "void main() {}",
            "shaderControls": {"width": 2},
            "mode2d": "lines_and_points",
            "mode3d": "lines",
            "lineWidth2d": 1.0,
            "lineWidth3d": 4.0,
        }

    def test_dict_is_coerced(self):
        layer = SegmentationLayer(
            source=SEG_SRC, skeleton_rendering={"mode_3d": "lines"}
        )
        assert isinstance(layer.skeleton_rendering, SkeletonRendering)
        assert layer.to_dict()["skeletonRendering"] == {"mode3d": "lines"}

    def test_invalid_mode(self):
        with pytest.raises(ValueError):
            SkeletonRendering(mode_3d="points")

    def test_shader_field_still_works(self):
        shader = DEFAULT_SHADER_MAP["skeleton_compartments"]
        d = SegmentationLayer(source=SEG_SRC).add_shader(shader).to_dict()
        assert d["skeletonRendering"] == {"shader": shader}

    def test_shader_combines_with_rendering(self):
        d = SegmentationLayer(
            source=SEG_SRC,
            shader="void main() {}",
            skeleton_rendering=SkeletonRendering(line_width_3d=2),
        ).to_dict()
        assert d["skeletonRendering"] == {
            "shader": "void main() {}",
            "lineWidth3d": 2.0,
        }

    def test_conflicting_shaders_raise(self):
        layer = SegmentationLayer(
            source=SEG_SRC,
            shader="void main() {}",
            skeleton_rendering=SkeletonRendering(shader="void main() { }"),
        )
        with pytest.raises(ValueError, match="skeleton shader"):
            layer.to_dict()


class TestAnnotationOptions:
    @pytest.fixture
    def local_kwargs(self):
        return dict(
            resolution=[4, 4, 40],
            annotations=[PointAnnotation(point=[1, 2, 3])],
            set_position=False,
        )

    def test_local_shader_controls(self, local_kwargs):
        d = AnnotationLayer(shader_controls={"size": 5}, **local_kwargs).to_dict()
        assert d["shaderControls"] == {"size": 5}

    def test_cloud_shader_controls(self):
        d = AnnotationLayer(source=ANNO_SRC, shader_controls={"size": 5}).to_dict()
        assert d["shaderControls"] == {"size": 5}

    def test_ignore_null_segment_filter(self, local_kwargs):
        d = AnnotationLayer(ignore_null_segment_filter=False, **local_kwargs).to_dict()
        assert d["ignoreNullSegmentFilter"] is False
        cloud = AnnotationLayer(source=ANNO_SRC, ignore_null_segment_filter=False)
        assert cloud.to_dict()["ignoreNullSegmentFilter"] is False

    def test_unset_is_omitted(self, local_kwargs):
        d = AnnotationLayer(**local_kwargs).to_dict()
        assert "shaderControls" not in d
        assert "ignoreNullSegmentFilter" not in d

    def test_shader_controls_survive_datamap(self):
        df = pd.DataFrame({"x": [1], "y": [2], "z": [3]})
        layer = AnnotationLayer(
            resolution=[4, 4, 40], set_position=False, shader_controls={"size": 3}
        ).add_points(DataMap("pts"), point_column=["x", "y", "z"])
        d = layer.map({"pts": df}).to_dict()
        assert d["shaderControls"] == {"size": 3}
        assert len(d["annotations"]) == 1
