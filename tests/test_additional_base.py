from unittest.mock import Mock, patch

import numpy as np
import pytest

from nglui.statebuilder.base import ViewerState
from nglui.statebuilder.ngl_components import (
    AnnotationLayer,
    CoordSpace,
    ImageLayer,
    SegmentationLayer,
)


class TestAdditionalBase:
    """Additional edge case tests for ViewerState to improve coverage"""

    def test_viewer_with_base_state(self):
        base_state = {"crossSectionScale": 2.0, "layout": "xy"}
        vs = ViewerState(base_state=base_state)
        assert vs.base_state == base_state

    def test_viewer_with_client_parameter(self):
        mock_client = Mock()
        vs = ViewerState(client=mock_client)
        assert vs._client is mock_client

    def test_add_layers_from_client_with_existing_client(self):
        mock_client = Mock()
        mock_client.info.viewer_site.return_value = "https://neuroglancer.com"
        mock_client.info.viewer_resolution.return_value = [4, 4, 40]
        mock_client.info.image_source.return_value = "precomputed://img"
        mock_client.info.segmentation_source.return_value = "precomputed://seg"
        mock_client.info.get_datastack_info.return_value = {"skeleton_source": None}

        vs = ViewerState(client=mock_client)
        result = vs.add_layers_from_client()  # Should use the stored client

        assert len(vs.layers) == 2

    def test_add_layers_from_client_imagery_only(self):
        mock_client = Mock()
        mock_client.info.viewer_site.return_value = "https://neuroglancer.com"
        mock_client.info.viewer_resolution.return_value = [4, 4, 40]
        mock_client.info.image_source.return_value = "precomputed://img"

        vs = ViewerState()
        vs.add_layers_from_client(mock_client, imagery=True, segmentation=False)

        assert len(vs.layers) == 1
        assert isinstance(vs.layers[0], ImageLayer)

    def test_add_layers_from_client_segmentation_only(self):
        mock_client = Mock()
        mock_client.info.viewer_site.return_value = "https://neuroglancer.com"
        mock_client.info.viewer_resolution.return_value = [4, 4, 40]
        mock_client.info.segmentation_source.return_value = "precomputed://seg"
        mock_client.info.get_datastack_info.return_value = {"skeleton_source": None}

        vs = ViewerState()
        vs.add_layers_from_client(mock_client, imagery=False, segmentation=True)

        assert len(vs.layers) == 1
        assert isinstance(vs.layers[0], SegmentationLayer)

    def test_add_annotation_layer_with_all_options(self):
        vs = ViewerState(dimensions=[4, 4, 40])
        result = vs.add_annotation_layer(
            name="complex_anno",
            source="precomputed://annotations",
            tags=["tag1", "tag2"],
            linked_segmentation="seg_layer",
            shader="annotation_shader",
        )

        layer = vs.layers[0]
        assert layer.name == "complex_anno"
        assert layer.source == "precomputed://annotations"
        assert layer.tags == ["tag1", "tag2"]
        assert layer.linked_segmentation == "seg_layer"
        assert layer.shader == "annotation_shader"
        assert result is vs

    def test_add_annotation_layer_with_linked_segmentation_dict(self):
        vs = ViewerState(dimensions=[4, 4, 40])
        linked_seg = {"segments": "seg_layer", "meshes": "mesh_layer"}

        vs.add_annotation_layer(name="anno_with_dict", linked_segmentation=linked_seg)

        layer = vs.layers[0]
        assert layer.linked_segmentation == linked_seg

    def test_add_segmentation_layer_with_all_options(self):
        vs = ViewerState()
        result = vs.add_segmentation_layer(
            source="precomputed://seg",
            name="complex_seg",
            resolution=[4, 4, 40],
            segments=[123, 456],
            selected_alpha=0.3,
            alpha_3d=0.7,
            mesh_silhouette=0.1,
        )

        layer = vs.layers[0]
        assert layer.name == "complex_seg"
        assert layer.selected_alpha == 0.3
        assert layer.alpha_3d == 0.7
        assert layer.mesh_silhouette == 0.1
        assert result is vs

    def test_add_image_layer_with_resolution_from_viewer(self):
        vs = ViewerState(dimensions=[8, 8, 80])
        vs.add_image_layer(source="precomputed://img", name="test_img")

        layer = vs.layers[0]
        assert layer.resolution == [8, 8, 80]

    def test_set_viewer_properties_with_coordspace_dimensions(self):
        vs = ViewerState()
        coord_space = CoordSpace(resolution=[4, 4, 40], units="μm")

        vs.set_viewer_properties(dimensions=coord_space)
        assert vs.dimensions is coord_space

    def test_viewer_property_setter_triggers_reset(self):
        vs = ViewerState()

        # Mock _reset_viewer to verify it's called
        with patch.object(vs, "_reset_viewer") as mock_reset:
            vs.dimensions = [4, 4, 40]
            mock_reset.assert_called_once()

        with patch.object(vs, "_reset_viewer") as mock_reset:
            vs.layout = "4panel"
            mock_reset.assert_called_once()

    def test_infer_coordinates_setter_triggers_reset(self):
        vs = ViewerState(infer_coordinates=True)

        with patch.object(vs, "_reset_viewer") as mock_reset:
            vs.infer_coordinates = False  # Change value
            mock_reset.assert_called_once()

        with patch.object(vs, "_reset_viewer") as mock_reset:
            vs.infer_coordinates = False  # Same value, should not trigger reset
            mock_reset.assert_not_called()

    def test_interactive_setter_resets_viewer(self):
        vs = ViewerState(interactive=False)

        # Set _viewer to something to test it gets reset
        vs._viewer = Mock()
        vs.interactive = True
        assert vs._viewer is None

    def test_add_annotation_source(self):
        vs = ViewerState()
        result = vs.add_annotation_source(
            source="precomputed://annotations",
            name="annotation_source",
            linked_segmentation="seg_layer",
            shader="test_shader",
        )

        layer = vs.layers[0]
        assert layer.name == "annotation_source"
        assert layer.source == "precomputed://annotations"
        assert layer.linked_segmentation == "seg_layer"
        assert layer.shader == "test_shader"
        assert result is vs

    @patch("nglui.statebuilder.source_info.populate_info")
    def test_source_info_caching(self, mock_populate_info):
        mock_populate_info.return_value = {"cached": "info"}

        vs = ViewerState()
        vs.add_layer(ImageLayer(name="test", source="precomputed://example"))

        # First access should populate
        info1 = vs.source_info
        assert info1 == {"cached": "info"}
        mock_populate_info.assert_called_once()

        # Second access should use cache
        info2 = vs.source_info
        assert info2 is info1
        mock_populate_info.assert_called_once()  # Still only called once

    def test_to_link_with_custom_text(self):
        vs = ViewerState()

        with patch.object(vs, "to_url") as mock_to_url:
            mock_to_url.return_value = "https://neuroglancer.com/test"

            link = vs.to_link(link_text="Custom Link Text")

            assert "Custom Link Text" in link.data
            assert "https://neuroglancer.com/test" in link.data

    def test_to_link_with_shorten_option(self):
        vs = ViewerState()

        with patch.object(vs, "to_url") as mock_to_url:
            mock_to_url.return_value = "https://neuroglancer.com/test"

            link = vs.to_link(shorten=True)

            mock_to_url.assert_called_with(
                target_url=None, target_site=None, shorten=True, client=None
            )

    def test_to_json_string_with_custom_indent(self):
        vs = ViewerState(dimensions=[4, 4, 40])

        with patch.object(vs, "to_dict") as mock_to_dict:
            mock_to_dict.return_value = {"test": "data"}

            json_str = vs.to_json_string(indent=8)

            # Should have large indentation
            assert "        " in json_str  # 8 spaces

    def test_viewer_property_lazy_evaluation(self):
        vs = ViewerState()

        # Mock to_neuroglancer_state
        with patch.object(vs, "to_neuroglancer_state") as mock_to_ng:
            mock_ng_state = Mock()
            mock_to_ng.return_value = mock_ng_state

            # First access
            viewer1 = vs.viewer
            assert viewer1 is mock_ng_state
            mock_to_ng.assert_called_once()

            # Second access should use cache
            viewer2 = vs.viewer
            assert viewer2 is viewer1
            mock_to_ng.assert_called_once()  # Still only called once

    def test_selected_layer_property_structure(self):
        vs = ViewerState()
        vs._selected_layer = "test_layer"
        vs._selected_layer_visible = True

        selected = vs.selected_layer
        assert selected == {"layer": "test_layer", "visible": True}

    def test_with_datamap_none(self):
        vs = ViewerState()

        with vs.with_datamap(None) as mapped_vs:
            assert mapped_vs is vs  # Should return self when datamap is None

    def test_with_datamap_non_dict(self):
        vs = ViewerState()
        simple_data = "test_data"

        with vs.with_datamap(simple_data) as mapped_vs:
            # Should convert non-dict to {None: data}
            assert mapped_vs is not vs  # Should be a copy

    @patch("nglui.statebuilder.source_info.suggest_position")
    def test_suggest_position_with_resolution(self, mock_suggest_position):
        mock_suggest_position.return_value = np.array([100, 200, 300])

        vs = ViewerState()
        position = vs._suggest_position_from_source(resolution=[4, 4, 40])

        mock_suggest_position.assert_called_once_with(vs.source_info, [4, 4, 40])
